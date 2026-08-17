"""Red neuronal híbrida cuántico-clásica para clasificación de MRI (TASK-10 / A3)."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.models.backbones import build_backbone
from src.models.heads import CabeceraReduccion
from src.models.vqc import circuito_vqc, forma_pesos_vqc, inicializar_pesos


def _inicializar_pesos_torch(tensor: torch.Tensor, n_capas: int, n_qubits: int) -> torch.Tensor:
    """Inicializa un tensor de pesos del VQC con la escala controlada de TASK-9."""
    with torch.no_grad():
        tensor.copy_(inicializar_pesos(n_capas, n_qubits))
    return tensor


class HQCNN(nn.Module):
    """Red neuronal híbrida cuántico-clásica para clasificación de MRI cerebral.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración del experimento con ``n_qubits`` y ``n_capas``.
    backbone : str
        Identificador del extractor congelado (por defecto ``efficientnet_b0``).

    Notes
    -----
    Sobrescribe ``train`` para que el backbone congelado permanezca en modo
    evaluación: de lo contrario las capas de normalización por lotes seguirían
    actualizando sus estadísticas móviles y el extractor congelado cambiaría
    de comportamiento entre épocas.

    La salida de la capa densa se acota con ``tanh(x) * π`` antes del
    ``AngleEmbedding`` porque los ángulos son periódicos módulo $2π$: sin
    acotación, dos características muy distintas pueden codificarse en el mismo
    estado cuántico sin ningún error visible.

    Los logits provienen de ``expval(Z)`` y viven en $[-1, 1]$ por construcción,
    lo que achata el *softmax* y limita la confianza alcanzable. Es una
    propiedad del diseño, no un error de entrenamiento.

    Con ``parameter-shift``, PennyLane no admite retropropagación sobre lotes
    en el QNode (issue #4462). El ``forward`` itera muestra a muestra sobre el
    ``TorchLayer`` y apila los resultados.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        backbone: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        self._n_capas_vqc = cfg.n_capas
        self.backbone, dim_latente = build_backbone(backbone)
        self.reduccion = CabeceraReduccion(dim_latente, cfg.n_qubits)

        qml.qnn.TorchLayer.set_input_argument("entradas")
        forma_pesos = {"pesos": forma_pesos_vqc(cfg.n_capas, cfg.n_qubits)}
        self.vqc = qml.qnn.TorchLayer(
            circuito_vqc,
            forma_pesos,
            init_method=lambda tensor: _inicializar_pesos_torch(
                tensor,
                cfg.n_capas,
                cfg.n_qubits,
            ),
        )

    @property
    def n_capas_vqc(self) -> int:
        """Profundidad ``L`` del ansatz; consumida por el ``Trainer`` vía duck typing."""
        return self._n_capas_vqc

    def train(self, modo: bool = True) -> HQCNN:
        """Cambia de modo manteniendo el backbone congelado en evaluación."""
        super().train(modo)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga de la imagen a los 4 logits.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de forma ``(B, 3, 224, 224)`` o ``(3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Logits de forma ``(B, 4)`` o ``(4,)`` con valores en $[-1, 1]$.
        """
        unica_muestra = x.ndim == 3
        if unica_muestra:
            x = x.unsqueeze(0)

        with torch.no_grad():
            latente = self.backbone(x)
        angulos = torch.tanh(self.reduccion(latente)) * torch.pi

        if unica_muestra:
            return self.vqc(angulos[0])

        return torch.stack(
            [self.vqc(angulos[i]) for i in range(angulos.shape[0])],
            dim=0,
        )


def contar_parametros_por_bloque(modelo: HQCNN) -> dict[str, int]:
    """Cuenta parámetros congelados y entrenables por bloque del HQCNN.

    Parameters
    ----------
    modelo : HQCNN
        Instancia del modelo híbrido.

    Returns
    -------
    dict[str, int]
        Claves ``backbone_congelado``, ``reduccion`` y ``vqc``.
    """
    backbone_congelado = sum(
        p.numel() for p in modelo.backbone.parameters() if not p.requires_grad
    )
    reduccion = sum(p.numel() for p in modelo.reduccion.parameters() if p.requires_grad)
    vqc = sum(p.numel() for p in modelo.vqc.parameters() if p.requires_grad)
    return {
        "backbone_congelado": backbone_congelado,
        "reduccion": reduccion,
        "vqc": vqc,
    }


def verificar_gradiente(
    modelo: HQCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-4,
    rtol: float = 0.15,
) -> tuple[float, float]:
    """Compara el gradiente analítico con diferencias finitas centradas.

    Parameters
    ----------
    modelo : HQCNN
        Modelo híbrido bajo prueba.
    x : torch.Tensor
        Entrada de imagen con forma ``(B, 3, 224, 224)``.
    y : torch.Tensor
        Etiquetas de clase con forma ``(B,)``.
    eps : float
        Perturbación para diferencias finitas centradas.
    rtol : float
        Tolerancia relativa declarada para la comparación en pruebas.

    Returns
    -------
    tuple[float, float]
        Gradiente analítico y numérico del peso del VQC con mayor magnitud.

    Notes
    -----
    La verificación se ejecuta en CPU. Con ``float32`` y ``parameter-shift`` la
    comparación es inestable; la tolerancia ``rtol`` es deliberadamente laxa para
    evitar falsos negativos intermitentes.
    """
    dispositivo_original = next(modelo.parameters()).device

    modelo_cpu = modelo.to(device=torch.device("cpu"))
    x_cpu = x.to(device=torch.device("cpu"), dtype=torch.float32)
    y_cpu = y.to(device=torch.device("cpu"))

    criterio = nn.CrossEntropyLoss()
    parametro = modelo_cpu.vqc.pesos

    modelo_cpu.zero_grad()
    perdida = criterio(modelo_cpu(x_cpu), y_cpu)
    perdida.backward()

    gradientes = parametro.grad.flatten()
    indice = int(gradientes.abs().argmax().item())
    analitico = float(gradientes[indice])

    with torch.no_grad():
        plano = parametro.flatten()
        original = float(plano[indice].item())
        plano[indice] = original + eps
        mas = float(criterio(modelo_cpu(x_cpu), y_cpu))
        plano[indice] = original - eps
        menos = float(criterio(modelo_cpu(x_cpu), y_cpu))
        plano[indice] = original

    numerico = (mas - menos) / (2 * eps)

    modelo.to(device=dispositivo_original)

    return analitico, numerico

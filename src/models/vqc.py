"""Clasificador cuántico variacional de 4 qubits (TASK-9 / A2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml
import torch

from src.logging.records import CLASES_ORDEN
from src.utils.seed import set_seed

N_QUBITS: int = 4
ESCALA_INICIALIZACION: float = 0.01
PROFUNDIDADES_DIAGNOSTICO: tuple[int, ...] = (2, 4, 6)
RUTA_DIAGRAMA_CIRCUITO: Path = Path("docs/trabajo_de_grado/Figuras/circuito_vqc.png")

MAPEO_QUBIT_CLASE: dict[int, str] = dict(enumerate(CLASES_ORDEN))

dispositivo_cuantico = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dispositivo_cuantico, interface="torch", diff_method="parameter-shift")
def circuito_vqc(entradas: torch.Tensor, pesos: torch.Tensor) -> list[torch.Tensor]:
    """Clasificador cuántico variacional de 4 qubits.

    Parameters
    ----------
    entradas : torch.Tensor
        Vector de 4 características acotadas provenientes de la capa densa.
    pesos : torch.Tensor
        Pesos variacionales de ``StronglyEntanglingLayers``.

    Returns
    -------
    list[torch.Tensor]
        Valor esperado del observable Z en cada qubit, uno por clase.

    Notes
    -----
    Se emplean mediciones locales, un observable por qubit, en lugar de un
    observable global: los costos globales inducen mesetas áridas incluso a
    profundidad baja, mientras que los locales preservan gradientes medibles
    (``Cerezo2021bp``). La codificación por ángulos define el mapa de
    características cuántico (``schuld2019feature``); con 4 características y
    4 qubits el circuito permanece superficial, requisito práctico en la era
    NISQ (``Preskill2018``).
    """
    qml.AngleEmbedding(entradas, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(pesos, wires=range(N_QUBITS))
    return [qml.expval(qml.Z(i)) for i in range(N_QUBITS)]


def forma_pesos_vqc(n_capas: int, n_qubits: int = N_QUBITS) -> tuple[int, ...]:
    """Devuelve la forma canónica de los pesos del ansatz.

    Parameters
    ----------
    n_capas : int
        Profundidad ``L`` de ``StronglyEntanglingLayers``.
    n_qubits : int
        Número de qubits del circuito.

    Returns
    -------
    tuple[int, ...]
        Forma obtenida de ``StronglyEntanglingLayers.shape``.
    """
    return qml.StronglyEntanglingLayers.shape(n_layers=n_capas, n_wires=n_qubits)


def inicializar_pesos(
    n_capas: int,
    n_qubits: int = N_QUBITS,
    escala: float = ESCALA_INICIALIZACION,
) -> torch.Tensor:
    """Inicializa los pesos cerca de la identidad para mitigar mesetas áridas.

    Parameters
    ----------
    n_capas : int
        Profundidad ``L`` del ansatz.
    n_qubits : int
        Número de qubits del circuito.
    escala : float
        Varianza controlada alrededor de la identidad.

    Returns
    -------
    torch.Tensor
        Tensor de pesos variacionales listo para el QNode.

    Notes
    -----
    La forma se obtiene de ``StronglyEntanglingLayers.shape`` y no se escribe a
    mano. Una escala pequeña mitiga la meseta árida de inicialización
    aleatoria en circuitos profundos (``McClean2018B``, ``grant2019initialization``).
    """
    forma = forma_pesos_vqc(n_capas, n_qubits)
    return torch.randn(forma) * escala


def norma_gradiente_inicial(
    n_capas: int,
    n_muestras: int = 20,
    *,
    semilla: int | None = None,
) -> float:
    """Mide la norma media del gradiente en la inicialización.

    Parameters
    ----------
    n_capas : int
        Profundidad ``L`` del ansatz.
    n_muestras : int
        Número de muestras aleatorias para promediar la norma.
    semilla : int | None
        Semilla opcional para reproducibilidad del diagnóstico.

    Returns
    -------
    float
        Norma media de ``‖∇pesos‖`` tras ``backward()``.

    Notes
    -----
    Un valor que decae con la profundidad es evidencia de meseta árida
    (``holmes2022connecting``).
    """
    if semilla is not None:
        set_seed(semilla)

    normas: list[float] = []
    for _ in range(n_muestras):
        entradas = torch.rand(N_QUBITS) * torch.pi
        pesos = inicializar_pesos(n_capas).requires_grad_(True)
        salida = torch.stack(circuito_vqc(entradas, pesos)).sum()
        salida.backward()
        normas.append(float(pesos.grad.norm()))
    return sum(normas) / len(normas)


def diagnosticar_normas_gradiente(
    profundidades: tuple[int, ...] = PROFUNDIDADES_DIAGNOSTICO,
    n_muestras: int = 20,
    semilla: int = 42,
) -> dict[int, float]:
    """Calcula la norma de gradiente inicial para varias profundidades.

    Parameters
    ----------
    profundidades : tuple[int, ...]
        Valores de ``L`` a diagnosticar antes de la ablación (TASK-11).
    n_muestras : int
        Muestras por profundidad.
    semilla : int
        Semilla fija para resultados reproducibles en la bitácora.

    Returns
    -------
    dict[int, float]
        Mapeo ``L → norma media del gradiente``.
    """
    return {
        n_capas: norma_gradiente_inicial(n_capas, n_muestras=n_muestras, semilla=semilla)
        for n_capas in profundidades
    }


def generar_diagrama_circuito(
    ruta: Path = RUTA_DIAGRAMA_CIRCUITO,
    n_capas: int = 4,
) -> Path:
    """Genera y guarda el diagrama del circuito con ``qml.draw_mpl``.

    Parameters
    ----------
    ruta : Path
        Ruta de salida del archivo PNG.
    n_capas : int
        Profundidad ``L`` usada en el diagrama ilustrativo.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura, _ = qml.draw_mpl(circuito_vqc)(torch.zeros(N_QUBITS), inicializar_pesos(n_capas))
    figura.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(figura)
    return ruta


def main() -> None:
    """Ejecuta el diagnóstico de gradiente y genera la figura del circuito."""
    normas = diagnosticar_normas_gradiente()
    for profundidad, norma in normas.items():
        print(f"L={profundidad}: norma_gradiente_inicial={norma:.6f}")

    ruta = generar_diagrama_circuito()
    print(f"Diagrama guardado en {ruta}")


if __name__ == "__main__":
    main()

"""Configuración centralizada del experimento."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Única fuente de verdad de hiperparámetros y rutas del experimento.

    Attributes
    ----------
    modelo : str
        Identificador que consume la fábrica de modelos.
    n_clases : int
        Número de clases del problema de clasificación.
    n_qubits : int
        Número de qubits del VQC; coincide con el número de clases.
    n_capas : int
        Profundidad ``L`` de las StronglyEntanglingLayers.
    data_fraction : float
        Fracción del entrenamiento usada en el escenario de escasez.
    n_folds : int
        Número de folds para validación cruzada estratificada.
    epocas : int
        Número de épocas de entrenamiento por corrida.
    batch_size : int
        Tamaño de lote para el DataLoader.
    lr : float
        Tasa de aprendizaje del optimizador.
    semilla : int
        Semilla global para reproducibilidad.
    raiz_datos : Path
        Raíz del conjunto de datos MRI.
    raiz_resultados : Path
        Raíz de métricas, historiales y artefactos tabulares.
    raiz_modelos : Path
        Raíz de pesos guardados (``state_dict``).
    raiz_figuras : Path
        Raíz de figuras y fragmentos LaTeX generados.
    """

    modelo: str = "hqcnn"
    n_clases: int = 4
    n_qubits: int = 4
    n_capas: int = 4
    data_fraction: float = 1.0
    n_folds: int = 5
    epocas: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    semilla: int = 42
    raiz_datos: Path = Path("data/brain_tumor_mri")
    raiz_resultados: Path = Path("results")
    raiz_modelos: Path = Path("models")
    raiz_figuras: Path = Path("results/figures")

    def ensure_layout(self) -> None:
        """Crea las carpetas de datos, resultados, modelos y figuras si no existen."""
        for ruta in (
            self.raiz_datos,
            self.raiz_resultados,
            self.raiz_modelos,
            self.raiz_figuras,
        ):
            ruta.mkdir(parents=True, exist_ok=True)

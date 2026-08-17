"""Bucle unificado de entrenamiento y evaluación (TASK-8)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.logging.records import (
    CLASES_ORDEN,
    EpochRecord,
    RunRecord,
    especificidad_por_clase,
    sensibilidad_por_clase,
)
from src.logging.sinks import corrida_existe, obtener_commit_sha
from src.logging.timing import medir_inferencia_ms_por_lote, sincronizar_dispositivo


def nombre_pesos(modelo: str, data_fraction: float, fold: int, semilla: int) -> str:
    """Construye el nombre de archivo para un ``state_dict`` persistido.

    Parameters
    ----------
    modelo : str
        Identificador del modelo.
    data_fraction : float
        Fracción de escasez del entrenamiento.
    fold : int
        Índice del fold estratificado.
    semilla : int
        Semilla global de la corrida.

    Returns
    -------
    str
        Nombre con convención ``{modelo}_{frac}_f{fold}_s{semilla}.pt``.
    """
    fraccion = str(data_fraction).replace(".", "p")
    return f"{modelo}_{fraccion}_f{fold}_s{semilla}.pt"


class Trainer:
    """Bucle único de entrenamiento y evaluación para modelos clásicos e híbridos.

    Parameters
    ----------
    modelo : nn.Module
        Modelo ya construido por la fábrica.
    cfg : ExperimentConfig
        Configuración del experimento.
    dispositivo : torch.device
        Dispositivo de cómputo seleccionado.
    fold : int
        Índice del fold estratificado de la corrida.

    Notes
    -----
    No importa ``pennylane`` ni ``torchvision``: recibe un ``nn.Module`` y una
    configuración, de modo que añadir un modelo nuevo no exige modificar esta
    clase (OCP) y el modelo híbrido es sustituible por uno clásico (LSP).
    """

    def __init__(
        self,
        modelo: nn.Module,
        cfg: ExperimentConfig,
        dispositivo: torch.device,
        *,
        fold: int,
    ) -> None:
        self._modelo = modelo.to(dispositivo)
        self._cfg = cfg
        self._dispositivo = dispositivo
        self._fold = fold
        self._criterio = nn.CrossEntropyLoss()
        parametros = [p for p in self._modelo.parameters() if p.requires_grad]
        self._optimizador = torch.optim.Adam(parametros, lr=cfg.lr)
        self._ultimo_registro: RunRecord | None = None

    def corrida_completada(self) -> bool:
        """Indica si la celda del diseño factorial ya está en ``experiments.csv``.

        Returns
        -------
        bool
            ``True`` si existe una fila con la misma tupla
            ``(modelo, data_fraction, fold, semilla)``.
        """
        registro = self._registro_identidad()
        ruta_csv = self._cfg.raiz_resultados / "experiments.csv"
        return corrida_existe(registro, ruta_csv)

    def ajustar(
        self,
        cargador_train: DataLoader,
        cargador_val: DataLoader,
    ) -> tuple[RunRecord, list[EpochRecord]]:
        """Entrena por un presupuesto fijo de épocas y evalúa al final.

        Parameters
        ----------
        cargador_train : DataLoader
            Cargador de entrenamiento.
        cargador_val : DataLoader
            Cargador de validación.

        Returns
        -------
        tuple[RunRecord, list[EpochRecord]]
            El registro de la corrida y su historial por época.

        Notes
        -----
        No se selecciona la mejor época mirando el conjunto de validación que
        luego se reporta: eso produciría una estimación optimista. Se usa el
        presupuesto completo de épocas, idéntico para todos los modelos.
        """
        historial: list[EpochRecord] = []
        inicio = time.perf_counter()

        for epoca in range(self._cfg.epocas):
            metricas_train = self._epoca_entrenamiento(cargador_train)
            metricas_val = self._evaluar_epoca(cargador_val)
            historial.append(
                EpochRecord(
                    epoca=epoca,
                    loss_train=metricas_train["loss_train"],
                    loss_val=metricas_val["loss_val"],
                    accuracy_train=metricas_train["accuracy_train"],
                    accuracy_val=metricas_val["accuracy_val"],
                )
            )

        sincronizar_dispositivo(self._dispositivo)
        tiempo_entrenamiento = time.perf_counter() - inicio

        metricas_train_final = self._evaluar_completo(cargador_train)
        metricas_val_final = self._evaluar_completo(cargador_val)
        inferencia_ms = medir_inferencia_ms_por_lote(
            self._modelo,
            cargador_val,
            self._dispositivo,
        )

        registro = self._construir_registro(
            metricas_train=metricas_train_final,
            metricas_val=metricas_val_final,
            tiempo_entrenamiento=tiempo_entrenamiento,
            inferencia_ms=inferencia_ms,
            n_train=len(cargador_train.dataset),
            n_val=len(cargador_val.dataset),
        )
        self._ultimo_registro = registro
        return registro, historial

    def guardar_pesos(self) -> Path:
        """Persiste los pesos del modelo con ``state_dict()`` exclusivamente.

        Returns
        -------
        Path
            Ruta del archivo ``.pt`` escrito en ``cfg.raiz_modelos``.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado ``ajustar()`` previamente.
        """
        if self._ultimo_registro is None:
            raise RuntimeError("Ejecuta ajustar() antes de guardar_pesos().")

        self._cfg.ensure_layout()
        nombre = nombre_pesos(
            self._cfg.modelo,
            self._cfg.data_fraction,
            self._fold,
            self._cfg.semilla,
        )
        ruta = self._cfg.raiz_modelos / nombre
        torch.save(self._modelo.state_dict(), ruta)
        return ruta

    def cargar_pesos(self, ruta: Path) -> None:
        """Recarga pesos desde un archivo ``state_dict`` persistido.

        Parameters
        ----------
        ruta : Path
            Ruta al archivo ``.pt`` generado por ``guardar_pesos()``.
        """
        estado = torch.load(ruta, map_location=self._dispositivo, weights_only=True)
        self._modelo.load_state_dict(estado)

    def _registro_identidad(self) -> RunRecord:
        """Construye un registro mínimo válido para consultar reanudabilidad."""
        return RunRecord(
            modelo=self._cfg.modelo,
            data_fraction=self._cfg.data_fraction,
            fold=self._fold,
            semilla=self._cfg.semilla,
            dispositivo=self._dispositivo.type,
            n_train=0,
            n_val=0,
            epocas=self._cfg.epocas,
            n_params_entrenables=0,
            n_capas_vqc=getattr(self._modelo, "n_capas_vqc", None),
            commit_sha=obtener_commit_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            accuracy_train=0.0,
            accuracy_val=0.0,
            loss_train=0.0,
            loss_val=0.0,
            f1_val_weighted=0.0,
            f1_val_macro=0.0,
            sensibilidad_por_clase={clase: 0.0 for clase in CLASES_ORDEN},
            especificidad_por_clase={clase: 0.0 for clase in CLASES_ORDEN},
            train_time_s=0.0,
            inference_ms_per_batch=0.0,
        )

    def _epoca_entrenamiento(self, cargador: DataLoader) -> dict[str, float]:
        """Ejecuta una época de entrenamiento y devuelve pérdida y exactitud."""
        self._modelo.train()
        perdida_total = 0.0
        correctos = 0
        total = 0

        for entradas, etiquetas in cargador:
            entradas = entradas.to(self._dispositivo)
            etiquetas = etiquetas.to(self._dispositivo)
            self._optimizador.zero_grad()
            logits = self._modelo(entradas)
            perdida = self._criterio(logits, etiquetas)
            perdida.backward()
            self._optimizador.step()

            lote = entradas.size(0)
            perdida_total += float(perdida.item()) * lote
            correctos += int((logits.argmax(dim=1) == etiquetas).sum().item())
            total += lote

        if total == 0:
            raise ValueError("El cargador de entrenamiento no produjo ningún lote.")

        return {
            "loss_train": perdida_total / total,
            "accuracy_train": correctos / total,
        }

    def _evaluar_epoca(self, cargador: DataLoader) -> dict[str, float]:
        """Evalúa pérdida y exactitud en validación para el historial por época."""
        self._modelo.eval()
        perdida_total = 0.0
        correctos = 0
        total = 0

        with torch.inference_mode():
            for entradas, etiquetas in cargador:
                entradas = entradas.to(self._dispositivo)
                etiquetas = etiquetas.to(self._dispositivo)
                logits = self._modelo(entradas)
                perdida = self._criterio(logits, etiquetas)

                lote = entradas.size(0)
                perdida_total += float(perdida.item()) * lote
                correctos += int((logits.argmax(dim=1) == etiquetas).sum().item())
                total += lote

        if total == 0:
            raise ValueError("El cargador de validación no produjo ningún lote.")

        return {
            "loss_val": perdida_total / total,
            "accuracy_val": correctos / total,
        }

    def _evaluar_completo(self, cargador: DataLoader) -> dict[str, float | dict[str, float]]:
        """Calcula métricas completas para el ``RunRecord`` final."""
        self._modelo.eval()
        perdida_total = 0.0
        total = 0
        verdaderos: list[int] = []
        predichos: list[int] = []

        with torch.inference_mode():
            for entradas, etiquetas in cargador:
                entradas = entradas.to(self._dispositivo)
                etiquetas = etiquetas.to(self._dispositivo)
                logits = self._modelo(entradas)
                perdida = self._criterio(logits, etiquetas)

                lote = entradas.size(0)
                perdida_total += float(perdida.item()) * lote
                total += lote
                verdaderos.extend(etiquetas.cpu().tolist())
                predichos.extend(logits.argmax(dim=1).cpu().tolist())

        if total == 0:
            raise ValueError("El cargador no produjo ningún lote para evaluación completa.")

        y_verdadero = np.array(verdaderos, dtype=np.int64)
        y_predicho = np.array(predichos, dtype=np.int64)
        matriz = confusion_matrix(
            y_verdadero,
            y_predicho,
            labels=list(range(len(CLASES_ORDEN))),
        )

        return {
            "loss": perdida_total / total,
            "accuracy": float((y_verdadero == y_predicho).mean()),
            "f1_weighted": float(
                f1_score(y_verdadero, y_predicho, average="weighted", zero_division=0.0)
            ),
            "f1_macro": float(
                f1_score(y_verdadero, y_predicho, average="macro", zero_division=0.0)
            ),
            "sensibilidad": sensibilidad_por_clase(y_verdadero, y_predicho),
            "especificidad": especificidad_por_clase(matriz),
        }

    def _construir_registro(
        self,
        *,
        metricas_train: dict[str, float | dict[str, float]],
        metricas_val: dict[str, float | dict[str, float]],
        tiempo_entrenamiento: float,
        inferencia_ms: float,
        n_train: int,
        n_val: int,
    ) -> RunRecord:
        """Ensambla el ``RunRecord`` validado a partir de métricas finales."""
        n_params = sum(
            p.numel() for p in self._modelo.parameters() if p.requires_grad
        )
        return RunRecord(
            modelo=self._cfg.modelo,
            data_fraction=self._cfg.data_fraction,
            fold=self._fold,
            semilla=self._cfg.semilla,
            dispositivo=self._dispositivo.type,
            n_train=n_train,
            n_val=n_val,
            epocas=self._cfg.epocas,
            n_params_entrenables=n_params,
            n_capas_vqc=getattr(self._modelo, "n_capas_vqc", None),
            commit_sha=obtener_commit_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            accuracy_train=float(metricas_train["accuracy"]),
            accuracy_val=float(metricas_val["accuracy"]),
            loss_train=float(metricas_train["loss"]),
            loss_val=float(metricas_val["loss"]),
            f1_val_weighted=float(metricas_val["f1_weighted"]),
            f1_val_macro=float(metricas_val["f1_macro"]),
            sensibilidad_por_clase=metricas_val["sensibilidad"],  # type: ignore[arg-type]
            especificidad_por_clase=metricas_val["especificidad"],  # type: ignore[arg-type]
            train_time_s=tiempo_entrenamiento,
            inference_ms_per_batch=inferencia_ms,
        )

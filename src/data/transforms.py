"""Fábrica de transformaciones de preprocesamiento y aumento (TASK-5 / A5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

TAMANO_ENTRADA = 224
MEDIA_IMAGENET: tuple[float, float, float] = (0.485, 0.456, 0.406)
DESV_IMAGENET: tuple[float, float, float] = (0.229, 0.224, 0.225)


def construir_transformaciones(*, aumentar: bool, tamano: int = TAMANO_ENTRADA) -> v2.Compose:
    """Devuelve el pipeline de transformaciones para una partición.

    Parameters
    ----------
    aumentar : bool
        ``True`` solo para la partición de entrenamiento.
    tamano : int
        Lado del cuadrado de entrada esperado por el backbone.

    Returns
    -------
    v2.Compose
        Pipeline de transformaciones torchvision v2.

    Notes
    -----
    La normalización usa las estadísticas de ImageNet porque los pesos del
    backbone fueron calibrados con ellas; usar las del propio dataset degrada
    las características transferidas.
    """
    pasos: list[v2.Transform] = [v2.Resize((tamano, tamano))]
    if aumentar:
        pasos += [
            v2.RandomRotation(degrees=10, fill=0),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    pasos += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEDIA_IMAGENET, std=DESV_IMAGENET),
    ]
    return v2.Compose(pasos)


def _seleccionar_ejemplos(manifiesto: pd.DataFrame) -> pd.DataFrame:
    """Selecciona hasta dos imágenes representativas (L y RGB si existen)."""
    ejemplos: list[pd.Series] = []
    for modo in ("L", "RGB"):
        candidatas = manifiesto[manifiesto["modo"] == modo]
        if not candidatas.empty:
            ejemplos.append(candidatas.iloc[0])
    if not ejemplos:
        ejemplos.append(manifiesto.iloc[0])
    return pd.DataFrame(ejemplos).reset_index(drop=True)


def _tensor_a_imagen(tensor: torch.Tensor) -> object:
    """Desnormaliza un tensor ImageNet para visualización."""
    muestra = tensor.detach().cpu().clone()
    for canal in range(3):
        muestra[canal] = muestra[canal] * DESV_IMAGENET[canal] + MEDIA_IMAGENET[canal]
    return muestra.clamp(0, 1).permute(1, 2, 0).numpy()


def generar_figura_aumento(
    manifiesto: pd.DataFrame,
    raiz: Path,
    ruta_salida: Path,
    *,
    semilla: int = 42,
) -> None:
    """Genera una figura con ejemplos antes y después del aumento.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Subconjunto del manifiesto con columnas ``ruta_relativa`` y ``modo``.
    raiz : Path
        Raíz del dataset.
    ruta_salida : Path
        Ruta de salida del PNG.
    semilla : int
        Semilla para reproducir las variantes aumentadas en la figura.
    """
    ejemplos = _seleccionar_ejemplos(manifiesto)
    transform_det = construir_transformaciones(aumentar=False)
    transform_aug = construir_transformaciones(aumentar=True)

    torch.manual_seed(semilla)
    n_filas = len(ejemplos)
    fig, ejes = plt.subplots(n_filas, 3, figsize=(9, 3 * n_filas), squeeze=False)

    titulos = ["Original", "Sin aumento (val)", "Con aumento (train)"]
    for col, titulo in enumerate(titulos):
        ejes[0, col].set_title(titulo)

    for fila_idx, (_, fila) in enumerate(ejemplos.iterrows()):
        with Image.open(raiz / fila["ruta_relativa"]) as img:
            imagen = img.convert("RGB")

        variantes = [
            imagen,
            transform_det(imagen),
            transform_aug(imagen),
        ]
        for col_idx, tensor_o_img in enumerate(variantes):
            eje = ejes[fila_idx, col_idx]
            if isinstance(tensor_o_img, torch.Tensor):
                eje.imshow(_tensor_a_imagen(tensor_o_img))
            else:
                eje.imshow(tensor_o_img)
            eje.axis("off")
            if col_idx == 0:
                eje.set_ylabel(f"{fila['modo']} — {fila['clase']}", fontsize=9)

    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)

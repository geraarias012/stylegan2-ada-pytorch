# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def load_labels(json_path):
    """Carga etiquetas desde un archivo JSON y las convierte en listas de números."""
    labels_dict = {}

    if json_path and isinstance(json_path, str) and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verificar si el JSON tiene la clave "labels"
        if isinstance(data, dict) and "labels" in data:
            raw_labels = data["labels"]

            # Convertir diccionarios en listas de valores numéricos
            labels_dict = {img: list(attrs.values()) if isinstance(attrs, dict) else attrs for img, attrs in raw_labels.items()}
            
            print(f"JSON cargado correctamente con {len(labels_dict)} imágenes etiquetadas.")
        else:
            print("Advertencia: No se encontró la clave 'labels' en el JSON o su estructura es incorrecta.")

    return labels_dict

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int], json_path=None):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    labels_dict = load_labels(json_path)

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            label = labels_dict.get(os.path.basename(fname), [])
            yield dict(img=img, label=label)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--json-file', help='Optional JSON file with labels', required=False, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)

def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    json_file: Optional[str],  # Antes: json -> Ahora: json_file
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch, including optional labels."""

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_image_folder(source, max_images=max_images, json_path=json_file)
    os.makedirs(dest, exist_ok=True)

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = PIL.Image.fromarray(image['img'])
        # Crear el directorio si no existe
        output_dir = os.path.dirname(os.path.join(dest, archive_fname))
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar la imagen
        img.save(os.path.join(dest, archive_fname), format='png')


        labels.append([archive_fname, image['label']] if image['label'] else None)

    metadata = { 'labels': labels if all(x is not None for x in labels) else None }
    with open(os.path.join(dest, 'dataset.json'), 'w') as f:
        json.dump(metadata, f)

    print("Dataset convertido con éxito.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset()

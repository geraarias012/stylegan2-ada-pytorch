import os
import re
import click
import torch
import dnnlib
import legacy
import numpy as np
from torchvision.utils import save_image

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=str, help='List of random seeds (e.g. 1,2,5-10)', required=True)
@click.option('--class', 'class_vector', type=str, help='Comma-separated class vector', required=False)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1.0, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
def generate_images(network_pkl, seeds, class_vector, truncation_psi, outdir):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    seed_list = []
    for s in seeds.split(','):
        if '-' in s:
            start, end = s.split('-')
            seed_list.extend(range(int(start), int(end) + 1))
        else:
            seed_list.append(int(s))

    if class_vector is not None:
        class_array = np.array([int(x) for x in class_vector.split(',')], dtype=np.float32)
        class_tensor = torch.tensor(class_array).to(device)
    else:
        class_tensor = None

    for seed in seed_list:
        print(f'Generating image for seed {seed} ...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        label = class_tensor.unsqueeze(0) if class_tensor is not None else torch.zeros([1, G.c_dim], device=device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
        img = (img + 1) * 0.5
        save_image(img, os.path.join(outdir, f'seed{seed:04d}.png'))

if __name__ == "__main__":
    generate_images()

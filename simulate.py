#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import importlib
import numpy as np
from skimage.io import imsave
from argparse import ArgumentParser, Namespace
from time import time
from types import ModuleType
import DRO

def save_image(npy, filename, upscale=None):
    npy -= np.amin(npy)

    if np.amax(npy) > 0.0:
        npy /= np.amax(npy)

    npy *= 255

    if upscale is not None:
        npy = npy.repeat(upscale, axis=0).repeat(upscale, axis=1)

    imsave(filename, np.transpose(npy.astype(np.uint8)))

def load_config(config_path: str) -> ModuleType:
    config_path = config_path.replace('/', '.')
    config = importlib.import_module(config_path)
    return config

def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="absolute path to the config file",
                        required=True)
    args = parser.parse_args()
    return args

def main():
    import sys
    print(sys.path)
    args = get_cli_arguments()
    config = load_config(config_path=args.config_path)
    dro_name = os.path.basename(args.config_path.replace('.', '/'))
    matrix = config.matrix
    fov = config.fov
    obj = config.obj
    voxel_size = fov / matrix

    print('  FOV / [mm]        = [%f, %f]' % (fov[0], fov[1]))
    print('  number of voxels  = [%d, %d]' % (matrix[0], matrix[1]))
    print('  voxel size / [mm] = [%f, %f]' % (voxel_size[0], voxel_size[1]))
    sys.stdout.flush()

    # simulation
    simulation_start = time()

    # construct k-space for DRO
    print('  simulation...')
    kspace = obj.kspace(fov, matrix)

    # save DRO to numpy file (k-space)
    # np.save(f'{dro_name}_Fourier_%dx%d.npy' % (matrix[0], matrix[1]), kspace)

    # add noise
    if config.noise is not None and config.noise != 0.0:
        noise_real = np.random.normal(scale=config.noise, size=kspace.shape)
        noise_imag = np.random.normal(scale=config.noise, size=kspace.shape)
        kspace += 1.0 * noise_real + 1.0j * noise_imag

    # inverse FFT
    img = DRO.kspace_to_image(kspace)

    # save DRO to numpy file
    np.save(f'{dro_name}_%dx%d.npy' % (matrix[0], matrix[1]), img)

    # save DRO to png
    save_image(np.abs(img), f'{dro_name}_%dx%d.png' %
               (matrix[0], matrix[1]))

    simulation_time = round(time() - simulation_start, 1)

    # Print only is simulation really happened (usually takes > 1 sec)
    if simulation_time > 1:
        print('simulation took:', simulation_time, 'sec.')

if __name__ == '__main__':
    main()
#!/usr/bin/python
import sys
import os
import copy
import subprocess
import argparse
import urllib.request
from vit_jax import flags

def progress( block_count, block_size, total_size ):
    ''' コールバック関数 '''
    percentage = 100.0 * block_count * block_size / total_size
    sys.stdout.write( "%.2f %% ( %d KB )\r"
        % ( percentage, total_size / 1024 ) )

if __name__ == '__main__':
    args = copy.copy(sys.argv)
    args.pop(0)
    tmp = ["/usr/bin/python","-m","vit_jax.train"]
    tmp.extend(args)
    args = tmp
    parser = flags.argparser(['ViT-B_16','ViT-B_32','ViT-L_16','ViT-L_32','ViT-H_14','testing'],
        ['cifar10','cifar100','imagenet2012'])
    parse_args = parser.parse_args()
    model_path= f'{parse_args.vit_pretrained_dir}/{parse_args.model}.npz'
    if not os.path.isfile(model_path):
        urllib.request.urlretrieve(url=f'https://storage.googleapis.com/vit_models/imagenet21k/{parse_args.model}.npz',filename=model_path,reporthook = progress)
    ret = subprocess.run(args).returncode
    sys.exit(ret)

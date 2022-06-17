"""Reference train script"""
import os

import ignite.distributed as idist


ROOT_PATH = os.getcwd()
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')
EXPERIMENT_NAME = ''
EXPERIMENT_PATH = os.path.join(CONFIG_PATH, f'{EXPERIMENT_NAME}.yaml')

def training(local_rank, config):

    # Data transformations (Augmentation)


    # Datasets


    # Data pipeline (Data loaders)


    # Model


    # Optim


    # Criterion





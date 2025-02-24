"""
Module with utility functions
"""
from rdkit import Chem
import os
import numpy as np

def canonicalize_smiles(sml):
    """
    Function that canonicalize a given SMILES
    :param sml: input SMILES
    :return: The canonical version of the input SMILES
    """
    mol = Chem.MolFromSmiles(sml)
    if mol is not None:
        sml = Chem.MolToSmiles(mol)
    return sml

def read_model(model_dir):
    """
    Read the model and max_min file from the given directory.

    Args:
        model_dir (str): Path to the directory containing the model and max_min file.

    Returns:
        tuple: A tuple containing (model_dir, x_max, x_min)
    """
    # Look for max_min file in the directory
    max_min_file = None
    for file in os.listdir(model_dir):
        if file.startswith('max_min') and (file.endswith('.npz') or file.endswith('.dat')):
            max_min_file = os.path.join(model_dir, file)
            break
    
    if max_min_file is None:
        raise FileNotFoundError(f"No max_min file found in {model_dir}")

    if max_min_file.endswith('.npz'):
        data = np.load(max_min_file)
        x_max = data['max'].astype('float32')
        x_min = data['min'].astype('float32')
    elif max_min_file.endswith('.dat'):
        data = np.loadtxt(max_min_file)
        x_max = data[0].astype('float32')
        x_min = data[1].astype('float32')
    else:
        raise ValueError(f"Unsupported file format: {max_min_file}")

    return model_dir, x_max, x_min
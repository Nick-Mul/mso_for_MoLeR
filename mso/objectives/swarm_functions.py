import requests
import datetime
import pandas as pd
import sys
import numpy as np
from scipy.interpolate  import interp1d
from rdkit import Chem
from functools import wraps
from mso.objectives.mol_functions import number_of_aromatic, penalize_long_aliphatic_chains, molecular_weight, tox_alert
import chemprop

def check_valid_mol(func):
    """
    Decorator function that checks if a mol object is None (resulting from a non-processable SMILES string)
    :param func: the function to decorate.
    :return: The decorated function.
    """
    @wraps(func)
    def wrapper(mol, *args, **kwargs):
        if mol is not None:
            return func(mol, *args, **kwargs)
        else:
            return 0
    return wrapper

def process_smi(smis):
    """function to convert list of smiles (i.e. swarm) into list of molecules, if error adds benzene to swarm

    Args:
        smis (_type_): list of smiles

    Returns:
        _type_: list of (rdkit) molecules
    """
    mols = []
    for smi in smis:
        try:
            mols.append(Chem.MolFromSmiles(smi))
        except:
            mols.append(Chem.MolFromSmiles("c1ccccc1"))
    return mols

def swarm_wt(smis):
    return [molecular_weight(m) for m in process_smi(smis)]

def swarm_number_aromatics(smis):
    return [number_of_aromatic(m) for m in process_smi(smis)]

def swarm_penalize_long_aliphatic_chains(smis, min_members):
    """
    Score that is 0 for molecules with aliphatic chains longer than min_members.
    """
    return [penalize_long_aliphatic_chains(m, min_members) for m in process_smi(smis)]

def toxic_swarm(smi):
    return [tox_alert(m) for m in process_smi(smi)]
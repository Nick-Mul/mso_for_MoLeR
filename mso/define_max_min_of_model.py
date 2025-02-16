from numpy import append
from molecule_generation import load_model_from_directory
import sys
import pandas as pd
import numpy as np
import click
import pickle


#### description ######

# while it's technically more elegant to define the boundaries of the system as a hypersphere what works here is to treat the system as a hypercube
# to calculate this we take the smiles list of the training set, encode this with model then use numpy to find the max and min of the vector.

smiles_file = sys.argv[1]
model_dir = sys.argv[2]

def calculate_max_min(smiles_file, model_dir):
    with open(smiles_file, 'r') as f: 
        smile_list = f.readlines() 

    with load_model_from_directory(model_dir) as model:
        embeddings = model.encode(smile_list)

    print("embedding complete")   
    stake = (np.vstack(embeddings))
    print("stake complete")
    out_max = np.max(stake,axis=0)
    out_min = np.min(stake,axis=0)

    np.savez('max_min.npz',max=out_max,min=out_min)


if __name__ == '__main__':
    calculate_max_min(smiles_file, model_dir)
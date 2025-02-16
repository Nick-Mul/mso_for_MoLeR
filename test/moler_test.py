import numpy as np
import pandas as pd
from molecule_generation import VaeWrapper
import sys

model_dir = sys.argv[1]
scaffold = "FC1=CC=CC(Cl)=C1" ## scaffold constraint all gen. chem. molecules have this  
init_mol = "O=CNC1=CC=CC=C1" 
moler_smiles = []
with VaeWrapper(model_dir) as model: ### makes 1000 random model like molecules
    [latent_center] = model.encode([init_mol])
    latents = latent_center + 0.5 * np.random.randn(10000, latent_center.shape[0]).astype(np.float32)
    for idx, smiles in enumerate(model.decode(latents, scaffolds=[scaffold] * len(latents))):
        print(f"Result #{idx + 1}: {smiles}")
        moler_smiles.append([smiles])
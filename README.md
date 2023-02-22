# mso_for_MoLeR
Modified Molecular Swarm Optimization (MSO) using MoLeR



This is a modifiation of Molecular Swarm optimisation (MSO) by Winter et al.,[1] to use the MoLeR model introduced in the paper "Learning to Extend Molecular Scaffolds with Structural Motifs" by Maziarz et al., [2] These modifications are described in appendix C of the paper.

install moler as described in their git

```
conda env create -f environment.yml
conda activate moler-env
pip install molecule-generation
```

The install this modification of winter's mso into the moler-env conda enviroment with as mso
```
git clone mso_for_MoLeR
cd mso_for_MoLeR
pip install -e .
```
Pretty much identical to Winter's example expect we using the MoLeR inference server.

```from numpy import append
import sys
import pandas as pd
from mso.optimizer import BasePSOptimizer
from mso.objectives.scoring import ScoringFunction
from mso.objectives.mol_functions import qed_score
from mso.moler_inference_server import _get_model_file, Inference_server


init_smiles = "OC(=O)C1=CC=CC=C1" # SMILES representation
scoring_functions = [ScoringFunction(func=qed_score, name="qed", is_mol_func=True)]

model_dir = _get_model_file(sys.argv[1])

print(model_dir)
inference_model = Inference_server(model_dir)
inference_model.__enter__()
embedding = inference_model.seq_to_emb([init_smiles])
print(embedding)
print(embedding[0].shape[0])

#error BasePSOptimizer not callable ??! for class method?! BasePSOptimizer.from_query. 
#opt = BasePSOptimizer(init_smiles=init_smiles,num_part=200,num_swarms=1,inference_model=inference_model,scoring_functions=scoring_functions)
opt = BasePSOptimizer.from_query(init_smiles=init_smiles,num_part=200,num_swarms=1,inference_model=inference_model,scoring_functions=scoring_functions)

opt(20)


inference_model.__exit__(None, None, None)
inference_model.__del__()
```



[1] Implementation of the method proposed in the paper "Efficient Multi-Objective Molecular Optimization in a Continuous Latent Space" by Robin Winter, Floriane Montanari, Andreas Steffen, Hans Briem, Frank Noé and Djork-Arné Clevert. Chemical Science, 2019, DOI: 10.1039/C9SC01928F https://pubs.rsc.org/en/content/articlelanding/2019/SC/C9SC01928F#!divAbstract

[2] Learning to Extend Molecular Scaffolds with Structural Motifs by Krzysztof Maziarz, Henry Jackson-Flux, Pashmina Cameron, Finton Sirockin, Nadine Schneider, Nikolaus Stiefl, Marwin Segler, Marc Brockschmidt https://arxiv.org/abs/2103.03864 

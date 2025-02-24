
from mso.optimiser import BasePSOptimizer
import numpy as np
from molecule_generation import VaeWrapper
from mso.objectives.swarm_functions import swarm_wt, swarm_number_aromatics, toxic_swarm 
from mso.objectives.scoring import SwarmScoringFunction, ScoringFunction
from mso.utils import read_model
import time
import os
import sys


init_smiles = "c1ccccc1" # SMILES representation
scaffold = "OC=O"
qsar_desirability = [{"x": 0.2, "y": 0.1}, {"x": 0.3, "y": 0.3}, {"x": 0.4, "y": 0.3}, {"x": 0.7, "y": 0.9}, {"x": 1, "y": 1}]
mwt_desirability = [{"x" : 180, "y" : 0}, {"x":200, "y": 1}, {"x":400, "y": 1.0}, {"x":450, "y": 0.0}]
scoring_functions = [SwarmScoringFunction(func=swarm_wt, name="mwt", desirability = mwt_desirability)]



    
model_dir, x_max, x_min = read_model(sys.argv[1])


with VaeWrapper(model_dir) as model:
    opt = BasePSOptimizer.from_query(init_smiles=init_smiles,num_part=500,num_swarms=2,inference_model=model,scoring_functions=scoring_functions, x_max=x_max, x_min=x_min, scaffold=scaffold)
    start_time = time.time()
    try:
        opt.run(20)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print("--- %s seconds ---" % (time.time() - start_time))
        try:
            opt.best_solutions.to_csv("best_solutions_" + timestr + ".csv", index=False)
            opt.best_fitness_history.to_csv("best_history_" + timestr + ".csv", index=False)
        except:
            pass
    except:
        opt.best_solutions.to_csv("best_solutions_" + "error" + ".csv", index=False)
        opt.best_fitness_history.to_csv("best_history_" + "error" + ".csv", index=False)

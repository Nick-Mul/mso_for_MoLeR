""""
Module defining the main Particle Swarm optimizer class.
"""
import time
import numpy as np
import logging
import multiprocessing as mp
import pandas as pd
from rdkit import Chem, rdBase
from mso.swarm import Swarm
from mso.utils import canonicalize_smiles
rdBase.DisableLog('rdApp.error')
logging.getLogger('tensorflow').disabled = True
from scipy.interpolate  import splrep

#modification of BasePSOptimiser to take smiles

class BasePSOptimizer:
    """
        Base particle swarm optimizer class. It handles the optimization of a swarm object.
    """
    def __init__(self, swarms, inference_model, scoring_functions=None):
        """

        :param swarms: List of swarm objects each defining an individual particle swarm that
            is used for optimization.
        :param inference_model: The inference model used to encode/decode pkl.
        :param scoring_functions: List of functions that are used to evaluate a generated molecule.
            Either take a RDKit mol object as input or a point in the cddd space.
        """
        self.inference_model = inference_model
        self.scoring_functions = scoring_functions
        self.swarms = swarms
        self.best_solutions = pd.DataFrame(columns=["smiles", "fitness"])
        self.best_fitness_history = pd.DataFrame(columns=["step", "swarm", "fitness"])

    def update_fitness(self, swarm):
        """
        Method that calculates and updates the fitness of each particle in  a given swarm. A
        particles fitness is defined as weighted average of each scoring functions output for
        this particle.
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        assert self.scoring_functions is not None
        weight_sum = 0
        fitness = 0
        #smile = [Chem.MolFromSmile swarm.smiles]
        for scoring_function in self.scoring_functions:
            unscaled_scores, scaled_scores, desirability_scores = scoring_function(swarm.smiles)
            swarm.unscaled_scores[scoring_function.name] = unscaled_scores
            swarm.scaled_scores[scoring_function.name] = scaled_scores
            swarm.desirability_scores[scoring_function.name] = desirability_scores
            fitness += scaled_scores
            weight_sum += scoring_function.weight
        fitness /= weight_sum
        swarm.update_fitness(fitness)
        return swarm

    def _next_step_and_evaluate(self, swarm):
        """
        Method that wraps the update of the particles position (next step) and the evaluation of
        the fitness at these new positions.
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        swarm.next_step()
        smiles = self.inference_model.decode(swarm.x, scaffolds=[swarm.scaffold]*len(swarm.x))
        swarm.smiles = smiles
        swarm.x = self.inference_model.encode(swarm.smiles)
        swarm = self.update_fitness(swarm)
        return swarm

    def _update_best_solutions(self, num_track):
        """
        Method that updates the best_solutions dataframe that keeps track of the overall best
        solutions over the course of the optimization.
        :param num_track: Length of the best_solutions dataframe.
        :return: The max, min and mean fitness of the best_solutions dataframe.
        """
        new_df = pd.DataFrame(columns=["smiles", "fitness"])
        new_df.smiles = [sml for swarm in self.swarms for sml in swarm.smiles]
        new_df.fitness = [fit for swarm in self.swarms for fit in swarm.fitness]
        new_df.smiles = new_df.smiles.map(canonicalize_smiles)
        #self.best_solutions = self.best_solutions.append(new_df)
        self.best_solutions = pd.concat([self.best_solutions, new_df])
        self.best_solutions = self.best_solutions.drop_duplicates("smiles")
        self.best_solutions = self.best_solutions.sort_values(
            "fitness",
            ascending=False).reset_index(drop=True)
        self.best_solutions = self.best_solutions.iloc[:num_track]
        best_solutions_max = self.best_solutions.fitness.max()
        best_solutions_min = self.best_solutions.fitness.min()
        best_solutions_mean = self.best_solutions.fitness.mean()
        return best_solutions_max, best_solutions_min, best_solutions_mean

    def _update_best_fitness_history(self, step):
        """
        tracks best solutions for each swarm
        :param step: The current iteration step of the optimizer.
        :return: None
        """
        new_df = pd.DataFrame(columns=["step", "swarm", "fitness", "smiles"])
        new_df.fitness = [swarm.swarm_best_fitness for swarm in self.swarms]
        new_df.smiles = [swarm.best_smiles for swarm in self.swarms]
        new_df.swarm = [i for i in range(len(self.swarms))]
        new_df.step = step
        #self.best_fitness_history = self.best_fitness_history.append(new_df, sort=False)
        self.best_fitness_history = pd.concat([self.best_fitness_history, new_df])


    def run(self, num_steps, num_track=10):
        """
        The main optimization loop.
        :param num_steps: The number of update steps.
        :param num_track: Number of best solutions to track.
        :return: The optimized particle swarm.
        """
        # evaluate initial score
        for swarm in self.swarms:
            self.update_fitness(swarm)
        for step in range(num_steps):
            self._update_best_fitness_history(step)
            max_fitness, min_fitness, mean_fitness = self._update_best_solutions(num_track)
            print("Step %d, max: %.3f, min: %.3f, mean: %.3f"
                  % (step, max_fitness, min_fitness, mean_fitness))
            for swarm in self.swarms:
                self._next_step_and_evaluate(swarm)
        return self.swarms

    @classmethod
    def from_query(cls, init_smiles, num_part, num_swarms, inference_model, x_min, x_max,
                   scoring_functions=None, phi1=2., phi2=2., phi3=2., v_min=-0.6, v_max=0.6, scaffold=None, **kwargs):
        """
        Classmethod to create a PSO instance with (possible) multiple swarms which particles are
        initialized at the position of the embedded input SMILES. All swarms are initialized at the
        same position.
        :param init_smiles: (string) The SMILES the defines the molecules which acts as starting
            point of the optimization. If it is a list of multiple smiles, num_part smiles will be randomly drawn.
        :param num_part: Number of particles in each swarm.
        :param num_swarms: Number of individual swarm to be optimized.
        :param inference_model: A inference model instance that is used for encoding an decoding
            SMILES to and from the CDDD space.
        :param scoring_functions: List of functions that are used to evaluate a generated molecule.
            Either take a RDKit mol object as input or a point in the cddd space.
        :param phi1: PSO hyperparamter.
        :param phi2: PSO hyperparamter.
        :param phi3: PSO hyperparamter.
        :param x_min: min bound of the optimization space (with CDDD this should be set to -1 as its the default
            take values between -1 and 1, with MoLeR -10). 
        :param x_max: max bound of the optimization space (with CDDD this should be set to +1 as its the default
            take values between -1 and 1, with MoLeR +10).
        :param v_min: minimal velocity component of a particle. Also used as lower bound for the
            uniform distribution used to sample the initial velocity.
        :param v_max: maximal velocity component of a particle. Also used as upper bound for the
            uniform distribution used to sample the initial velocity.
        :param kwargs: additional parameters for the PSO class
        :return: A PSOptimizer instance.
        """
        inference_model = inference_model
        embedding = inference_model.encode([init_smiles])
        swarms = [
            Swarm.from_query(
                init_sml=init_smiles,
                init_emb=embedding,
                num_part=num_part,
                v_min=v_min,
                v_max=v_max,
                x_min=x_min,
                x_max=x_max,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3,
                scaffold=scaffold) for _ in range(num_swarms)]
        return cls(swarms, inference_model, scoring_functions, **kwargs)
    

    @classmethod
    def swarms_from_smiles_list(cls, init_smiles, num_swarms, inference_model, x_min, x_max, v_min=-0.6, v_max=0.6, scoring_functions=None, phi1=2., phi2=2., scaffold=None, **kwargs):
        """Classmethodd to create a PSO instance with (possible) multiple swarms which particles are
        initialized at the position of the embedded input SMILES list. 

        Args:
            init_smiles (list of strings): list of smiles
            num_swarms (int): 
            inference_model (model): A inference model instance that is used for encoding an decoding SMILES to and from the latent space
            x_min (vector): min bound of the optimization space
            x_max (vector): min bound of the optimization space
            v_min (float, optional): minimal velocity component of a particle. Also used as lower bound for the
            uniform distribution used to sample the initial velocity. Defaults to -0.6.
            v_max (float, optional): maximal velocity component of a particle. Also used as upper bound for the
            uniform distribution used to sample the initial velocity. Defaults to 0.6.
            scoring_functions (_type_, optional): _description_. Defaults to None.
            phi1 (_type_, optional): PSO hyperparamter. Defaults to 2.
            phi2 (_type_, optional): PSO hyperparamter. Defaults to 2.
            scaffold (smiles, optional): smiles string. Defaults to None.
            param kwargs: additional parameters for the PSO class
       
        Returns:
            _type_: A PSOptimizer instance.
        """
        embedding = inference_model.seq_to_emb(init_smiles)
        v = np.random.uniform(v_min, v_max, [len(init_smiles), embedding[0].shape[0]])
        swarms = [Swarm(init_smiles,embedding, v=v, x_min=x_min, x_max=x_max, phi1=phi1,
                phi2=phi2, scaffold=scaffold) for _ in range(num_swarms)]

        return cls(swarms, inference_model, scoring_functions,**kwargs)
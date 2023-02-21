import pathlib
import random
from typing import ContextManager, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from rdkit import Chem
import sys
from molecule_generation.utils.moler_inference_server import MoLeRInferenceServer
from molecule_generation.utils.model_utils import get_model_class, get_model_parameters

Pathlike = Union[str, pathlib.Path]


def _get_model_file(dir: Pathlike) -> pathlib.Path:
    """Retrieves the MoLeR pickle file from a given directory.

        Args:
            dir: Directory from which the model should be retrieved.

        Returns:
            Path to the model pickle.

        Raises:
            ValueError, if the model pickle is not found or is not unique.
        """
        # Candidate files must end with ".pkl"
    candidates = list(pathlib.Path(dir).glob("*.pkl"))
    if len(candidates) != 1:
        raise ValueError(
            f"There must be exactly one *.pkl file. Found the following: {candidates}."
            )
    else:
        return candidates[0]

class Inference_server(ContextManager):
    def __init__(self, trained_model_path,number_workers = 6, beam_size = 1):
        self.trained_model_path = trained_model_path
        self.number_workers =number_workers
        self.beam_size = beam_size

    def __enter__(self):
        self._inference_server = MoLeRInferenceServer(self.trained_model_path, self.number_workers, max_num_samples_per_chunk=500 // self.beam_size)
        return self
    
    def seq_to_emb(self,smiles_list: List[str], include_log_variances: bool = False
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Encode input molecules to vectors in the latent space.

        Args:
            smiles_list: List of molecules as SMILES.
            include_log_variances: Whether to also return log variances on the latent encodings.

        Returns:
            List of results. Each result is the mean latent encoding if `include_log_variances` is
            `False`, and a pair containing the mean and the corresponding log variance otherwise.
        """
        # Note: if we ever start being strict about type hints, we could properly express the
        # relationship between `include_log_variances` and the return type using `@overload`.

        return self._inference_server.encode(
            smiles_list, include_log_variances=include_log_variances
        )
    
    def emb_to_seq(
        self,
        latents: List[np.ndarray],  # type: ignore
        scaffolds: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """Decode molecules from latent vectors, potentially conditioned on scaffolds.

        Args:
            latents: List of latent vectors to decode.
            scaffolds: List of scaffold molecules, one per each vector. Each scaffold in
                the list can be `None` (denoting lack of scaffold) or the whole list can
                be `None`, which is synonymous with `[None, ..., None]`.

        Returns:
            List of SMILES strings.
        """
        if scaffolds is not None:
            scaffolds = [
                Chem.MolFromSmiles(scaffold) if scaffold is not None else None
                for scaffold in scaffolds
            ]

        return [
            smiles_str
            for smiles_str, _ in self._inference_server.decode(
                latent_representations=np.stack(latents),
                include_latent_samples=False,
                init_mols=scaffolds,
                beam_size=self.beam_size,
            )
        ]

    
    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        # Standard Python convention, we can ignore the types
        inference_server = getattr(self, "_inference_server", None)
        if inference_server is not None:
            inference_server.__exit__(exc_type, exc_value, traceback)
            delattr(self, "_inference_server")

    def __del__(self):
        inference_server = getattr(self, "_inference_server", None)
        if inference_server is not None:
            inference_server.cleanup_workers()



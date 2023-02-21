from mso.moler_inference_server import Inference_server, _get_model_file
import sys


server = Inference_server(_get_model_file(sys.argv[1]))
server.__enter__()

init_mol = "COC(=O)C(C)N1N=C(C)C2=CC=C(OC3=C(F)C=C(C=C3Cl)C(F)(F)F)C=C12"
scaffold = "CC(O)=O"
vec = server.seq_to_emb([init_mol])
print(vec)
vec = server.seq_to_emb([scaffold])
print(vec)
smi = server.emb_to_seq(vec)
print(smi)
server.__exit__(None,None,None)
server.__del__()
# mso_for_MoLeR
Modified Molecular Swarm Optimization (MSO) using MoLeR



This is a modifiation of Molecular Swarm optimisation (MSO) by Winter et al.,[1] to use the MoLeR model introduced in the paper "Learning to Extend Molecular Scaffolds with Structural Motifs" by Maziarz et al., [2] These modifications are described in appendix C of the paper.

install moler as described in their git

```
conda env create -f environment.yml
conda activate moler-env
pip install molecule-generation
```

The package is installed into the moler-env conda enviroment with as mso
```
git clone mso_for_MoLeR
cd mso_for_MoLeR
pip install -e .
```



[1] Implementation of the method proposed in the paper "Efficient Multi-Objective Molecular Optimization in a Continuous Latent Space" by Robin Winter, Floriane Montanari, Andreas Steffen, Hans Briem, Frank Noé and Djork-Arné Clevert. Chemical Science, 2019, DOI: 10.1039/C9SC01928F https://pubs.rsc.org/en/content/articlelanding/2019/SC/C9SC01928F#!divAbstract

[2] Learning to Extend Molecular Scaffolds with Structural Motifs by Krzysztof Maziarz, Henry Jackson-Flux, Pashmina Cameron, Finton Sirockin, Nadine Schneider, Nikolaus Stiefl, Marwin Segler, Marc Brockschmidt https://arxiv.org/abs/2103.03864 

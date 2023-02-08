# Installation

## Pre-requisites:
1. Working [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) GPU runtime.
2. Conda environment with Rdkit installed

Then:
```bash
git clone https://github.com/andriusbern/vinaGPU && cd vinaGPU
pip install -e .

## Pull the docker image
sudo docker pull andriusbern/vina-gpu:latest
```

## Usage

## Python interface
```python
import os
from vinagpu import VinaGPU

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)
target_path = os.path.join('examples', 'P21918.pdb') 
active_site = (2.753, 0.994, -7.633) # Active site coordinates of P21918
output_subfolder = 'P21918_test' # results stored at: "./output/P21918_test"

smiles = [
    'NC[C@@H](NC(=O)c1ccc(F)cc1)C1CCCC1',
    'CCc1cccc(C(C)(C)NCCc2ccc(OC)c(OC)c2)c1',
    'CCN1CCCC1CNS(=O)(=O)c1ccc(F)cc1',
    'O=C(Cn1cccc1)Nc1ccc(Cl)cc1Cl']

vina_docker = VinaGPU()

scores = vina_docker.dock(
    target_pdb_path=target_path,
    smiles=smiles,
    output_subfolder=output_subfolder, 
    active_site_coords=active_site,
    verbose=True)
```

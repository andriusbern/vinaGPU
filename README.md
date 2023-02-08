

# Vina-GPU via docker

This package contains a minimalistic python API for running VinaGPU via a docker image.

# Installation

## 1. Pre-requisites:
1. Nvidia driver version>=515.43.04
2. Working [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) GPU runtime.
3. Python3.8+ conda environment with Rdkit installed
   
---

## 2. Install the package, dependencies: 

```bash
git clone https://github.com/andriusbern/vinaGPU && cd vinaGPU
pip install -e .
pip install meeko docker scipy
```

## 3. Pull the docker image

```
sudo docker pull andriusbern/vina-gpu:latest
```
The docker image contains:
- Cuda 11.7
- [Vina-GPU](https://github.com/DeltaGroupNJUPT/Vina-GPU) (compiled with boost 1.77.0, cuda 11.7, proper OpenCL dependencies)
- Protein preprocessing tools:
    - [ADFR Suite](https://ccsb.scripps.edu/adfr/downloads/)
    - [pdb_tools](https://wenmr.science.uu.nl/pdbtools/)

---

## Usage (python)

```python
import os
from vinagpu import VinaGPU

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)
target_path = os.path.join('examples', 'P21918.pdb') 
active_site = (2.753, 0.994, -7.633) # Active site coordinates of P21918
output_subfolder = 'P21918_test' # results stored at: "./P21918_test"

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

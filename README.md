

# Vina-GPU via docker

This package contains a minimalistic python API for running VinaGPU via a docker image.

## Features:

1. Can be used to dock on multiple GPUs, multiple workers per GPU.
2. CPU workers using AutoDock Vina python API can be run in parallel to the GPU workers

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
pip install meeko docker scipy dimorphite-dl vina
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

## Usage: 8 parallel GPU workers (on 4 GPUS) + 8 CPU workers (8 threads each)
```python
import time
from vinagpu import parallel_dock

target_pdb_path = 'examples/P21918.pdb'
output_subfolder = 'test_docking'

with open('examples/valid_smiles.txt', 'r') as f:
    smiles = f.read().splitlines()

t0 = time.time()

parallel_dock(target_pdb_path=target_pdb_path, smiles=smiles,      
              output_subfolder=output_subfolder,
              num_cpu_workers=8, exhaustiveness=8, threads_per_cpu_worker=8,
              gpu_ids=[0,1,2,3], workers_per_gpu=2)

t1 = time.time()
print('Ligands per second: {}'.format(len(smiles) / (t1 - t0)))
print('Total time: {}'.format(t1 - t0))
```

## Usage, single GPU worker

```python
import os
from vinagpu import VinaGPU

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)
target_pdb_path = os.path.join('examples', 'P21918.pdb') 
active_site = (2.753, 0.994, -7.633) # Active site coordinates of P21918
output_subfolder = 'P21918_test' # results stored at: "./P21918_test"

smiles = [
    'NC[C@@H](NC(=O)c1ccc(F)cc1)C1CCCC1',
    'CCc1cccc(C(C)(C)NCCc2ccc(OC)c(OC)c2)c1',
    'CCN1CCCC1CNS(=O)(=O)c1ccc(F)cc1',
    'O=C(Cn1cccc1)Nc1ccc(Cl)cc1Cl']

vina_docker = VinaGPU()

scores = vina_docker.dock(
    target_pdb_path=target_pdb_path,
    smiles=smiles,
    output_subfolder=output_subfolder, 
    active_site_coords=active_site,
    verbose=True)
```

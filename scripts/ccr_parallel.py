import os
from vinagpu import VinaGPU
import time
from vinagpu import parallel_dock

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)
target_pdb_path = os.path.join('examples', 'ccr.pdb') 
box_center = (5.1, 28, 187.6) # Active site coordinates of P21918
box_size   = (16.2, 17.8, 17.4)
output_subfolder = 'ccr_parallel_3workers_optimal' # results stored at: "./P21918_test"

with open('examples/SL_5000_10.csv') as f:
    smiles = f.readlines()
    print(smiles)
    smiles = [x.strip('\n') for x in smiles]
    smiles = [x.strip('"') for x in smiles]

smiles = smiles

# print(smiles)
t0 = time.time()

parallel_dock(target_pdb_path=target_pdb_path, 
              smiles=smiles[1:150],      
              box_center=box_center,
              box_size=box_size,
              output_subfolder=output_subfolder,
              num_cpu_workers=0, exhaustiveness=8, threads_per_cpu_worker=8, # CPU worker parameters
              gpu_ids=[1, 2, 3], workers_per_gpu=1, search_depth=9,
              threads=1024)          # GPU Worker parameters

t1 = time.time()
print(f'Docked ligands per second: {len(smiles) / (t1 - t0)}')
print(f'Total time: {t1 - t0}')


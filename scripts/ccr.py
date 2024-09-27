import os
from vinagpu import VinaGPU

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)
target_pdb_path = os.path.join('examples', 'ccr.pdb') 
box_center = (5.1, 28, 187.6) # Active site coordinates of P21918
box_size   = (16.2, 17.8, 17.4)
output_subfolder = 'ccr_t' # results stored at: "./P21918_test"

with open('examples/SL_5000_10.csv') as f:
    smiles = f.readlines()[:10]
    print(smiles)
    smiles = [x.strip('\n') for x in smiles]
    smiles = [x.strip('"') for x in smiles]

metadata = [{'a': 1, 'b': 2} for _ in range(len(smiles))]
print(smiles)

vina_docker = VinaGPU()

scores = vina_docker.dock(
    target_pdb_path=target_pdb_path,
    smiles=smiles,
    output_subfolder=output_subfolder, 
    box_center=box_center,
    box_size=box_size,
    verbose=True,
    write_log=True,
    clean=True,
    metadata=metadata)
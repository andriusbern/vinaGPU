import os
from vinagpu import VinaGPU
import pandas as pd

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)

generated_path = 'examples/SL_5000_10.csv'
generated_smiles = pd.read_csv(generated_path)

target_pdb_path = os.path.join('examples', 'ccr.pdb') 
box_center = (5.1, 28, 187.6) # Active site coordinates of P21918
box_size   = (16.2, 17.8, 17.4)
output_subfolder = 'TESTu' # results stored at: "./P21918_test"

vina_docker = VinaGPU()

df = vina_docker.dock_dataframe(
    dataframe=generated_smiles[:100],
    target_pdb_path=target_pdb_path,
    output_subfolder=output_subfolder, 
    box_center=box_center,
    box_size=box_size,
    verbose=True,
    write_log=True,
    clean=True)

print(df.columns)
print(df.head())

df.to_csv('examples/test.csv', index=False)
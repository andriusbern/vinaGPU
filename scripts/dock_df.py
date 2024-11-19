import os
import time
from vinagpu import VinaGPU
import pandas as pd
import argparse

# Docking example on P21918 (DRD5 - D(1B) dopamine receptor)

def main(gpu_id, output_folder, n_ligands):
    generated_path = 'examples/SL_5000_10.csv'
    generated_smiles = pd.read_csv(generated_path)

    target_pdb_path = os.path.join('examples', 'ccr.pdbqt')
    box_center = (5.1, 28, 187.6)  # Active site coordinates of P21918
    box_size   = (16.2, 17.8, 17.4)

    # Initialize VinaGPU with the specified GPU device
    vina_docker = VinaGPU(devices=[gpu_id])
    generated_smiles.head()

    t0 = time.time()
    df = vina_docker.dock_dataframe(
        dataframe=generated_smiles[:n_ligands],
        target_pdb_path=target_pdb_path,
        output_subfolder=output_folder, 
        box_center=box_center,
        box_size=box_size,
        verbose=True,
        write_log=True,
        threads=1024,
        clean=True)

    t1 = time.time()
    print(f'Docked ligands: {len(df)} in {t1 - t0} seconds')

    print(df.columns)
    print(df.head())
    df.to_csv('scripts/docking_results.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dock a list of ligands to a target protein using VinaGPU')
    parser.add_argument('--gpu_id', type=str, required=True, help='ID of the GPU to use for docking')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to store docking results')
    parser.add_argument('--n_ligands', type=int, required=True, help='Number of ligands to dock')
    args = parser.parse_args()

    main(args.gpu_id, args.output_folder, args.n_ligands)


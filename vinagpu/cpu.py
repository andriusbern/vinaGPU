# Import necessary modules for the class
import os
import shutil
import time
import datetime
import pandas as pd

import rdkit.Chem.GraphDescriptors
from vinagpu.base import BaseVinaRunner

from vina import Vina

from dimorphite_dl import DimorphiteDL


class VinaCPU(BaseVinaRunner):
    """
    Class for running docking simulations with CPU.
    The ligands will be prepared but the receptor should already be prepared. It also predicts the protomers
    and return the one with the best affinity.

    Methods:
        get_protomers:
            Finds the protomers (different protonation states) of a molecule.
        dock:
            Docks the prepared ligands using AutoDock Vina.
        prepare_ligand:
            Prepares the protomers for docking
    """

    def __init__(self, box_center=[0, 0, 0],
                 box_size=[0, 0, 0], exhaustiveness=8, n_poses=9, cpu=1, seed=0,
                 min_rmsd=1.0, docking_output_dir='docking', device_id=None,
                 mol_prepare_dir=None):

        super(VinaCPU, self).__init__(device='cpu')

        self.counter = 0
        self.device_id = device_id
        self.box_center = box_center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.min_rmsd = min_rmsd

        self.v = Vina(sf_name='vina', seed=seed, cpu=cpu, verbosity=0)
        self.mol_prepare_dir = mol_prepare_dir

        """ 
        Initialize the Docking class.
        Parameters
        -----------
            receptor_pdbqt (str): Path to the receptor PDBQT file.

            box_center(list of floats):
                Coordinates of the center of the search space.
            box_size (list of floats):
                Dimensions of the search space. 
            exhaustiveness (int):
                Exhaustiveness of the search, by default 8.
            n_poses (int):
                Maximum number of binding poses to output, by default 9.
            cpu (int):
                Number of CPUs to use, by default 1.
            seed (int):
                Seed for the random number generator, by default 0.
            min_rmsd (float):
                Minimum RMSD for pose clustering, by default 1.0.
            docking_output_dir(str):
                Output directory for docking results, by default 'docking'.
            mol_prepare_dir(str):
                Directory for molecule preparation, by default None.
        """

    def get_protomers(self, smiles, ph_range=(6, 7), max_variants=128, pka_precision=0.5):
        """
        Finds the protomers , which are different protonation states of the molecule
        Args:
            smiles(list): A list of SMILES strings.
            ph_range(tuple, optional): The pH range for protomer generation. Defaults to (6, 7).
            max_variants(int, optional): The maximum number of protomers to generate for each SMILES string. Defaults to 128.
            pka_precision(float, optional): The precision for pKa calculations. Defaults to 0.5.
        Returns:
             list: A list of protomers for each SMILES string.
        """
        dimorphite_dl = DimorphiteDL(
            min_ph=ph_range[0],
            max_ph=ph_range[1],
            max_variants=max_variants,
            label_states=False,
            pka_precision=pka_precision
        )
        protomers_list = []
        for smile in smiles:
            protomers = dimorphite_dl.protonate(smile)
            protomers_list.append(protomers)
        return protomers_list


    def dock(self, target_pdb_path, smiles=[], ligand_pdbqt_paths=[], output_subfolder='',
             box_center=(0,0,0), box_size=(20,20,20), exhaustiveness=5, **kwargs):
        """
        Dock a list of SMILES strings to the target protein using AutoDock Vina

        Args:
            smiles (list): A list of SMILES strings to be docked.

        Returns:
            list: A list of the best affinities (lowest energy) for each SMILES string.

        """

        results_path = os.path.join(self.out_path, output_subfolder)
        os.makedirs(results_path, exist_ok=True)

        protomers_list = self.get_protomers(smiles)
        scores = []

        # Prepare target
        if target_pdb_path.endswith('.pdb'): # If target is a .pdb file, convert to .pdbqt
            target_pdbqt_path = os.path.join(results_path, os.path.basename(target_pdb_path).replace('.pdb', '.pdbqt'))
            if not os.path.exists(target_pdbqt_path):
                target_pdbqt_path = self.prepare_target(target_pdb_path, out_path=results_path)
        else: # If target is already in .pdbqt format, just copy it to results_path
            target_pdbqt_path = os.path.join(results_path, os.path.basename(target_pdb_path))
            shutil.copyfile(target_pdb_path, target_pdbqt_path)

        self.v.set_receptor(target_pdbqt_path)
        self.v.compute_vina_maps(center=box_center, box_size=box_size)

        print('Docking ligands...')
        timing, dates = [], []
        for i, protomers in enumerate(protomers_list):
            t0 = time.time()
            best_affinity = None
            

            mol_id = f"docking_id_{self.counter}_ligand_{i}"
            out_prefix = f"{self.out_path}/pose_{mol_id}.best.out"
            for protomer in protomers:
                pdbqt_string = self.prepare_ligand(protomer)
                self.v.set_ligand_from_string(pdbqt_string)
                self.v.dock(exhaustiveness=exhaustiveness, n_poses=self.n_poses,
                            min_rmsd=self.min_rmsd, )
                energies = self.v.energies(n_poses=self.n_poses)
                # calculates the energy of the first pose
                best_energy = energies[0][0]
                if not best_affinity or best_affinity > best_energy:
                    # if best_affinity is greater than best_energy then the value of best_affinity is also best_energy
                    best_affinity = best_energy

                    self.v.write_poses(f"{out_prefix}.pdbqt",
                                       n_poses=self.n_poses, overwrite=True)
                    pd.DataFrame(energies, columns=["Total", "Inter", "Intra", "Torsions", "Intra_best_pose"]).to_csv(
                        f"{out_prefix}.tsv", sep="\t",
                        header=True, index=False)
                    with open(f"{out_prefix}.smi", "w", encoding="utf-8") as smi:
                        smi.write(protomer)
            scores.append(best_affinity)

            dates += [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            timing += [round(time.time() - t0, 2)]
            print(f'+ {self.device}:{self.device_id} | [{dates[-1]} | t={timing[-1]}s] Docked ligand {i+1}/{len(protomers_list)} | Affinity values: {scores[i]}...')
            # returns the best_affinity which is the lowest energy of all the poses
        return scores


    def prepare_ligand(self, protomer):
        """
        Prepare the ligand for docking by converting its SMILES string representation
        to a molecule, adding hydrogen atoms, embedding it in 3D space, and writing
        its PDBQT string representation.

        Arguments:
             protomer(str): SMILES string representation of the protomer.
        Returns:
            str: PDBQT string representation of the prepared ligand.

            """

        lig = rdkit.Chem.MolFromSmiles(protomer)
        protonated_lig = rdkit.Chem.AddHs(lig)
        rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)
        self.molecule_preparation.prepare(protonated_lig)

        return self.molecule_preparation.write_pdbqt_string()



if __name__ == "__main__":
    # define the docking parameters

    receptor_pdbqt = "examples/P21918.pdb"
    box_center = [0, 0, 0]
    box_size = [10, 10, 10]
    exhaustiveness = 8
    n_poses = 9
    cpu = 1
    seed = 0
    min_rmsd = 1.0
    docking_output_dir = "docking"

    smiles = []
    targets = ['P21918', 'C3SWJ7', 'O14757']
    for target in targets[2:]:
        smiles_path = f'/home/andrius/datasets/data_smiles/{target}/smiles.txt'
        with open(smiles_path, 'r') as f:
            smiles += f.read().splitlines()

    # initialize the docking class
    docking = VinaCPU(box_center=box_center,
                      box_size=box_size, exhaustiveness=exhaustiveness,
                      n_poses=n_poses, cpu=cpu, seed=seed, min_rmsd=min_rmsd)
                      
    docking.dock(
        target_pdb_path=receptor_pdbqt,
        smiles=smiles,
        output_subfolder=docking_output_dir,
        active_site_coords=box_center,
        bbox_size=box_size,
        exhaustiveness=exhaustiveness
    )
import os
import shutil
import subprocess as sp
from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem
import docker
from vinagpu.utils import run_executable


class BaseVinaRunner:
    """
    Class methods for running Vina-GPU docker container
    Also contains methods for preparing the ligand and target:
        - Ligand preparation via rdkit and meeko
        - Target preparation via ADFR Suite and pdb_tools
    """
    def __init__(self, device, adfr_suite_path=None, out_path=None):
        self.device = device
        self.device_id = None
        
        if out_path is None:
            path = os.getcwd()
            self.out_path = os.path.join(path, 'out')
        else:
            self.out_path = out_path

        self.adfr_suite_docker_path = '/htd/ADFRsuite-1.0'
        self.adfr_suite_path = adfr_suite_path # Local path to ADFR Suite (optional)
        self.vina_dir = '/vina-gpu-dockerized/Vina-GPU-2.1/QuickVina2-GPU-2.1'
        self.docking_dir = self.vina_dir + '/docking'
        self.molecule_preparation = MoleculePreparation(rigid_macrocycles=True)
        self.client = docker.from_env()
        self.container = None
        self.docker_kwargs = dict(
            image='vina',
            volumes = [f'{self.out_path}:{self.docking_dir}'])  


    def start_docker_container(self):
        """ 
        Start Vina-GPU docker container (runs until it is killed)
        Returns:
            docker container object
        """

        container = self.client.containers.run(
            command='sleep infinity', # Keeps the container running until it is killed
            detach=True,              # Run container in background
            **self.docker_kwargs)
        
        return container
 

    def remove_docker_container(self):
        """
        Stop Vina-GPU docker container
        """
        self.container.remove(force=True) 
        self.container = None
        

    @staticmethod
    def dock(self, target_pdb_path, smiles, out_path=None):
        """
        Dock the ligand to the target, return the docking scores

        Arguments:
            target_pdb_path (str) : path to the target .pdb file
            smiles (list)         : list of smiles strings
            out_path (str)        : path to save the .pdbqt file (default: ./drugex/utils/docking/output)
        Returns:
            list of docking scores
        """
        scores = [0]
        return scores

        
    def prepare_ligand(self, smiles, out_path=None):
        """
        Prepare ligand for docking, return ligand .pdbqt file path

        Arguments:
            smiles (str)     : smiles string
            out_path (str)   : path to save the .pdbqt file (default: ./drugex/utils/docking/output)
        Returns:
            path to the ligand .pdbqt file
        """
        try:
            # Ligand preparation via rdkit and meeko
            mol = Chem.MolFromSmiles(smiles)             # type: ignore
            protonated_ligand = Chem.AddHs(mol)          # type: ignore
            AllChem.EmbedMolecule(protonated_ligand)     # type: ignore
            self.molecule_preparation.prepare(protonated_ligand)

            # Write to .pdbqt file required by Vina
            if out_path is None:
                out_path = self.out_path
            self.molecule_preparation.write_pdbqt_file(out_path)
        except Exception as e:
            print(f'Error while preparing ligand: {e}')
            out_path = None
        return out_path


    def prepare_target(self, pdb_path, output_path=None, chain='A', use_docker=True):
        """ 
        TODO:
        1. Move this to the Protein class (maybe?)
        2. Would require a DockerContainer class to be created (to isolate Docker-related methods)

        To be used in the dock method if the target is not already prepared

        Prepare target for docking, return target pdbqt path
        Arguments:
            pdb_path (str)   : path to target .pdb file
            out_path (str)   : path to save the .pdbqt file
            chain (str)      : chain to use for docking (if target is a multi-chain protein)
            use_docker (bool): use docker container to prepare the target
        Returns:
            path to the processed target .pdbqt file
        """

        ## Output filenames

        extension = pdb_path.split('.')[-1]
        assert os.path.isfile(pdb_path), f'Invalid file path: {pdb_path}'
        assert extension in ['pdb', 'pdbqt'], f'Invalid file type: {extension}'

        if pdb_path.endswith('.pdbqt'): # If target is already in .pdbqt format, just copy it to the results_path
            target_pdbqt_path = os.path.join(output_path, os.path.basename(pdb_path))
            if not os.path.exists(target_pdbqt_path):
                shutil.copyfile(pdb_path, target_pdbqt_path)
            return target_pdbqt_path

        # Prepare target (if target is a .pdb file, convert to .pdbqt)
        target_pdbqt_path = os.path.join(output_path, os.path.basename(pdb_path).replace('.pdb', '.pdbqt'))
        if not os.path.isfile(target_pdbqt_path):
            if output_path is None:
                output_path = self.out_path
            basename = os.path.basename(pdb_path)
            out_file_path = os.path.join(output_path, basename)              # This is where the target .pdb file will be saved
            shutil.copyfile(pdb_path, out_file_path)                         # Copy target .pdb file to output folder   
            chain_basename = basename.replace('.pdb', f'_chain_{chain}.pdb') # Name of the .pdb file with only the selected chain
            chain_pdb_path = os.path.join(output_path, chain_basename)       # Full path to the .pdb file with only the selected chain
            pdbqt_basename = basename.replace('.pdb', '.pdbqt')              # Name of the .pdbqt file
            target_pdbqt_path = os.path.join(output_path, pdbqt_basename)    # Full path to the .pdbqt file

            print(f'Preparing {basename} for docking: selecting chain [{chain}] and creating {target_pdbqt_path} file...')

        if not use_docker: # Processing locally using ADFR Suite and pdb_tools
            cmd = f'pdb_selchain -{chain} {pdb_path} | pdb_delhetatm | \
                    pdb_tidy > {chain_pdb_path}'
            run_executable(cmd, shell=True)

            adfr_binary = os.path.join(self.adfr_suite_path, 'bin', 'prepare_receptor')
            cmd = f'{adfr_binary} -r {chain_pdb_path} \
                    -o {target_pdbqt_path} -A checkhydrogens'
            run_executable(cmd)
        
        else: # Processing within the docker container

            # Select a single chain in case the target is a multimer
            if self.container is None:
                self.container = self.start_docker_container()
            try:
                workdir = self.docking_dir + '/' + os.path.basename(output_path)
                print(workdir)
                cmd = f"bash -c 'pdb_selchain -{chain} {basename} | pdb_delhetatm | \
                        pdb_tidy > {chain_basename}'"
                self.container.exec_run(
                    cmd=cmd,
                    workdir=workdir,
                    demux=True)

                ## Prepare the target for docking using ADFR Suite 'prepare_receptor' binary
                adfr_binary = os.path.join(self.adfr_suite_path, 'bin', 'prepare_receptor')
                cmd = f'{adfr_binary} -r {chain_basename} -o {pdbqt_basename} -A checkhydrogens'
                self.container.exec_run(
                    cmd=cmd,
                    workdir=workdir,
                    demux=True)
            except Exception as e:
                print(f'Error while preparing target: {e}')
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
            finally:
                self.remove_docker_container()

        return target_pdbqt_path

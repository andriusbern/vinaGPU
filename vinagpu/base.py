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

        use_chain = False
        # Prepare target
        if pdb_path.endswith('.pdb'): # If target is a .pdb file, convert to .pdbqt
            filename_pdb = os.path.basename(pdb_path)
            filename_pdbqt = filename_pdb.replace('.pdb', '.pdbqt')
            filename_chain = filename_pdb.replace('.pdb', f'_chain_{chain}.pdb')
            input_pdb_path = os.path.join(output_path, filename_pdb)
            output_pdbqt_path = os.path.join(output_path, filename_pdbqt)

            if not os.path.isfile(output_pdbqt_path):
                if output_path is None:
                    output_path = self.out_path
                shutil.copyfile(pdb_path, os.path.join(output_path, filename_pdb)   )                         # Copy target .pdb file to output folder   
                chain_pdb_path = os.path.join(output_path, filename_chain)       # Full path to the .pdb file with only the selected chain
                # target_pdbqt_path = os.path.join(output_path, pdbqt_basename)    # Full path to the .pdbqt file

                print(f'Preparing {filename_pdb} for docking: selecting chain [{chain}] and creating {output_pdbqt_path} file...')

                if not use_docker: # Processing locally using ADFR Suite and pdb_tools
                    cmd = f'pdb_selchain -{chain} {input_pdb_path} | pdb_delhetatm | \
                            pdb_tidy > {chain_pdb_path}'
                    run_executable(cmd, shell=True)

                    adfr_binary = os.path.join(self.adfr_suite_path, 'bin', 'prepare_receptor')
                    cmd = f'{adfr_binary} -r {chain_pdb_path} \
                            -o {output_pdbqt_path} -A checkhydrogens'
                    run_executable(cmd)
                
                else: # Processing within the docker container
                    # Select a single chain in case the target is a multimer

                    if self.container is None:
                        self.container = self.start_docker_container()
                    try:
                        workdir = self.docking_dir + '/' + os.path.basename(output_path)
                        print(workdir)
                        ## list files in the workdir

                        if use_chain:

                            cmd = f"bash -c 'pdb_selchain -{chain} {filename_pdb} | pdb_delhetatm | \
                                    pdb_tidy > {filename_chain}'"
                            self.container.exec_run(
                                cmd=cmd,
                                workdir=workdir,
                                demux=True)

                            ## Prepare the target for docking using ADFR Suite 'prepare_receptor' binary
                            adfr_binary = os.path.join(self.adfr_suite_docker_path, 'bin', 'prepare_receptor')
                            cmd = f'{adfr_binary} -r {filename_chain} -o {filename_pdbqt} -A checkhydrogens'
                            _, (stdout, stderr) = self.container.exec_run(
                                cmd=cmd,
                                workdir=workdir,
                                demux=True)
                            print(stdout)
                            print(stderr)
                        else:


                            ## Prepare the target for docking using ADFR Suite 'prepare_receptor' binary
                            adfr_binary = os.path.join(self.adfr_suite_docker_path, 'bin', 'prepare_receptor')
                            cmd = f'{adfr_binary} -r {filename_pdb} -o {filename_pdbqt} -A checkhydrogens'
                            _, (stdout, stderr) = self.container.exec_run(
                                cmd=cmd,
                                workdir=workdir,
                                demux=True)
                            print(stdout)
                            print(stderr)
                        
                    except Exception as e:
                        print(f'Error while preparing target: {e}')
                    except KeyboardInterrupt:
                        print('KeyboardInterrupt')
                    finally:
                        self.remove_docker_container()

                # target_pdbqt_path = self.prepare_target(pdb_path, out_path=output_path)
        elif pdb_path.endswith('.pdbqt'): # If target is already in .pdbqt format, just copy it to the results_path
            # target_pdbqt_path = os.path.join(output_path, os.path.basename(pdb_path))
            if not os.path.exists(output_pdbqt_path):
                shutil.copyfile(pdb_path, output_pdbqt_path)
        else:
            # target_pdbqt_path = None
            raise ValueError(f'Invalid file type: {pdb_path}')

        return output_pdbqt_path


    def visualize_results(self, target_pdb_path, ligand_pdbqt_path, scores):
        """
        Displays all the docked ligands along with the target in PyMOL
        """
        # Check if pymol is installed
        try:
            import pymol
        except ImportError:
            print('PyMOL is not installed. Please install it to visualize the results.')
            return
        
        pymol.finish_launching(['pymol', '-q'])
        pymol.cmd.load(target_pdb_path, os.path.basename(target_pdb_path).split('.')[0])
        if type(ligand_pdbqt_path) is str:
            ligand_pdbqt_path = [ligand_pdbqt_path]
        for i, ligand in enumerate(ligand_pdbqt_path):
            try:
                pymol.cmd.load(ligand, f'ligand{i}{scores[i][0]}')
            except:
                pass
        
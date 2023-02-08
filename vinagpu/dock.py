import os, time, datetime, re
import shutil
import docker
import subprocess as sp
from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem


def run_executable(cmd, shell=True, **kwargs):
    """ Run executable command and return output from stdout and stderr """
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=shell, **kwargs)
    stdout, stderr = proc.communicate()
    return (stdout, stderr)

def process_stdout(stdout):
    """ Processes the stdout of Vina, returns the affinity of each docking orientation. """
    affinities = []
    is_int = re.compile(r'^\s*\d+\s*$')
    for line in stdout.splitlines():
        if bool(is_int.match(line.decode('utf-8')[:4])):
            orientation_id, affinity, dist1, dist2  = line.split()
            affinities += [float(affinity)]
    return affinities


class VinaGPU:
    """
    Class methods for running Vina-GPU docker container
    Also contains methods for preparing the ligand and target:
        - Ligand preparation via rdkit and meeko
        - Target preparation via ADFR Suite and pdb_tools
    """
    def __init__(self, docker_image_name='andriusbern/vina-gpu', devices=['all'], adfr_suite_path=''):
        path = os.path.dirname(os.path.abspath(__file__))
        self.out_path = os.path.join(path, 'output')
        self.vina_dir = '/vina-gpu-dockerized/vina'
        self.docking_dir = self.vina_dir + '/docking'
        self.adfr_suite_docker_path = '/htd/ADFRsuite-1.0'
        self.adfr_suite_path = adfr_suite_path # Local path to ADFR Suite
        self.client = docker.from_env()
        self.docker_image_name = docker_image_name
        self.molecule_preparation = MoleculePreparation()

        ## Configuration for running the Vina-GPU docker container 
        # (requires nvidia-docker runtime)
        dev_req = docker.types.DeviceRequest  # type: ignore
        self.docker_kwargs = dict(
            image=docker_image_name,
            remove=True,       # Remove container after execution
            runtime='nvidia',  # Use nvidia-docker runtime
            device_requests=[dev_req(device_ids=devices, capabilities=[['gpu']])])
        

    def dock(self, target_pdb_path, smiles=[], ligand_pdbqt_paths=[],
             output_subfolder='', active_site_coords=(0,0,0), bbox_size=(20,20,20), 
             threads=512, thread_per_call=256, clean=True, verbose=False, 
             search_depth=3,
             visualize_in_pymol=False, write_log=True):
        """
        Use Vina-GPU docker image to dock ligands (list of SMILES or .pdbqt files) to the target. 
        Produces a .pdbqt file for each ligand (with multiple docked orientations). 

        Arguments:
            target_pdb_path (str)                   : path to target pdb file
            smiles: (list(str))                     : list of smiles strings    
            ligand_pdbqt_paths (list(str))          : list of paths to ligand pdbqt files
            output_subfolder (str), opt             : subfolder to save output files
            active_site_coords (tuple(float)), opt  : coordinates of the active site of the target (x,y,z)=(0,0,0)
            bbox_size (tuple(float)), opt           : size of the bounding box around the active site (x,y,z)=(20,20,20)
            threads (int), opt                      : number of threads to use for docking
            thread_per_call (int), opt              : number of threads to use for each call to Vina
            clean (bool), opt                       : remove ligand .pdbqt files after docking
            verbose (bool), opt                     : print docking progress, scores, etc.
            visualize_in_pymol (bool), opt          : visualize the docking results in pymol
            write_log (bool), opt                   : write log file with docking results
        Returns:
            all_scores (list(list((float)))         : list of docking scores for each ligand
        """

        assert (len(ligand_pdbqt_paths) > 0) or (len(smiles) > 0), \
        "Either a list of ligand .pdbqt paths or a list of smiles strings must be provided"

        if len(smiles) > 0:
            ligand_pdbqt_paths = []

        results_path = os.path.join(self.out_path, output_subfolder)
        os.makedirs(results_path, exist_ok=True)

        ### Preprocessing
        # Prepare target
        if target_pdb_path.endswith('.pdb'): # If target is a .pdb file, convert to .pdbqt
            # Check if .pdbqt file already exists in results_path
            target_pdbqt_path = os.path.join(results_path, os.path.basename(target_pdb_path).replace('.pdb', '.pdbqt'))
            if not os.path.exists(target_pdbqt_path):
                target_pdbqt_path = self.prepare_target(target_pdb_path, out_path=results_path)
        else: # If target is already in .pdbqt format, just copy it to results_path
            target_pdbqt_path = os.path.join(results_path, os.path.basename(target_pdb_path))
            shutil.copyfile(target_pdb_path, target_pdbqt_path)

        # If no ligand .pdbqt paths are provided create them from smiles
        if len(ligand_pdbqt_paths) < 1:      
            for i, mol in enumerate(smiles):    
                ligand_pdbqt_path = os.path.join(results_path, f'ligand_{i}.pdbqt')
                out_path = self.prepare_ligand(mol, out_path=ligand_pdbqt_path)
                if out_path is not None:
                    ligand_pdbqt_paths.append(ligand_pdbqt_path)
        basenames = [os.path.basename(p) for p in ligand_pdbqt_paths]
        basenames_docked = [lig.replace('.pdbqt', '_docked.pdbqt') for lig in basenames]
        ligand_paths_docked = [os.path.join(results_path, p) for p in basenames_docked]
        
        ### Run Vina-GPU docker container
        all_scores, timing, dates = [], [], []
        t0_total = time.time()
        for i, ligand_filename in enumerate(basenames):
            t0 = time.time()

            cmd = f'./Vina-GPU  \
                --receptor docking/{os.path.basename(target_pdbqt_path)} \
                --ligand docking/{ligand_filename} \
                --out docking/{basenames_docked[i]} \
                --center_x {active_site_coords[0]} \
                --center_y {active_site_coords[1]} \
                --center_z {active_site_coords[2]} \
                --size_x {bbox_size[0]} \
                --size_y {bbox_size[1]} \
                --size_z {bbox_size[2]} \
                --thread {threads} \
                --search_depth {search_depth} \
                --thread_per_call {thread_per_call}'

            volumes = [f'{results_path}:{self.docking_dir}']
            stdout = self.client.containers.run(
                command=cmd,
                working_dir=self.vina_dir,
                volumes=volumes,
                **self.docker_kwargs)

            scores = process_stdout(stdout)
            all_scores += [scores]

            timing += [round(time.time() - t0, 2)]
            dates += [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            if verbose:
                print(f'- [{dates[-1]} | t={timing[-1]}s] Docked ligand {i+1}/{len(basenames)} | Affinity values: {scores}...')

        if write_log:
            log_path = os.path.join(results_path, 'log.tsv')
            header = 'date\ttarget\taffinity\tsmiles\ttime'
            write_header = os.path.exists(log_path)
            with open(log_path, 'a') as f:
                f.write(header + '\n') if write_header else None
                for i, scores in enumerate(all_scores):
                    f.write(f'{dates[i]}\t{os.path.basename(target_pdb_path)}\t{scores[0]}\t{smiles[i]}\t{timing[i]}\n')

        if visualize_in_pymol: 
            self.visualize_results(target_pdb_path, ligand_paths_docked, scores=all_scores)

        if clean: # Remove intermediate files (undocked ligand .pdbqt files)
            for path in ligand_pdbqt_paths:
                os.remove(path)
            for path in ligand_paths_docked:
                os.remove(path)
        
        print(f'Finished docking {len(smiles)} ligands in {round(time.time() - t0_total, 2)}s')
        print('Results saved in', results_path)
        return all_scores, ligand_paths_docked
        

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


    def prepare_target(self, pdb_path, out_path=None, chain='A', use_docker=True):
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
        if out_path is None:
            out_path = self.out_path
        basename = os.path.basename(pdb_path)
        out_file_path = os.path.join(out_path, basename)
        shutil.copyfile(pdb_path, out_file_path)
        chain_basename = basename.replace('.pdb', f'_chain_{chain}.pdb')
        chain_pdb_path = os.path.join(out_path, chain_basename)
        pdbqt_basename = basename.replace('.pdb', '.pdbqt')
        target_pdbqt_path = os.path.join(out_path, pdbqt_basename)

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
            volumes = [f'{out_path}:{self.docking_dir}']
            cmd = f"bash -c 'pdb_selchain -{chain} {basename} | pdb_delhetatm | \
                    pdb_tidy > {chain_basename}'"
            self.client.containers.run(
                command=cmd,
                working_dir=self.docking_dir,
                volumes=volumes,
                **self.docker_kwargs)

            ## Prepare the target for docking using ADFR Suite 'prepare_receptor' binary
            adfr_binary = os.path.join(self.adfr_suite_docker_path, 'bin', 'prepare_receptor')
            cmd = f'{adfr_binary} -r {chain_basename} -o {pdbqt_basename} -A checkhydrogens'
            self.client.containers.run(
                command=cmd,
                working_dir=self.docking_dir,
                volumes=volumes,
                **self.docker_kwargs)

        return target_pdbqt_path


    def visualize_results(self, target_pdb_path, ligand_pdbqt_path, scores):
        """ Displays all the docked ligands along with the target in PyMOL"""

        import pymol
        pymol.finish_launching(['pymol', '-q'])
        pymol.cmd.load(target_pdb_path, os.path.basename(target_pdb_path).split('.')[0])
        if type(ligand_pdbqt_path) is str:
            ligand_pdbqt_path = [ligand_pdbqt_path]
        for i, ligand in enumerate(ligand_pdbqt_path):
            pymol.cmd.load(ligand, f'ligand{i}{scores[i][0][0]}')
        
        
if __name__ == "__main__":

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
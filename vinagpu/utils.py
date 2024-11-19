from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import zlib
import re
import subprocess as sp
import os 
from collections import OrderedDict

def run_executable(cmd, shell=True, **kwargs):
    """ Run executable command and return output from stdout and stderr """
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=shell, **kwargs)
    stdout, stderr = proc.communicate()
    return (stdout, stderr)


def process_stdout(stdout):
    """ Processes the stdout of Vina, returns the affinity of each docking orientation. """
    affinities, buffer = [], []
    return_dict = OrderedDict()
    is_int = re.compile(r'^\s*\d+\s*$')
    for line in stdout.splitlines():

        if bool(is_int.match(line.decode('utf-8')[:4])):
            orientation_id, affinity, dist1, dist2  = line.split()
            buffer += [float(affinity)]

        if line.startswith(b'Writing'):
            ligand_id = line.split()[-2].decode('utf-8').split('/')[-1]
            affinities += [buffer]
            return_dict[ligand_id] = buffer
            buffer = []

    return affinities, return_dict


def partition_output(output_text):
    """
    Splits the docking output text into individual ligand result chunks.
    
    Parameters:
    output_text (str): The raw output text from the docking software.
    
    Returns:
    list of str: Each element is a chunk of text for one ligand.
    """
    ## Change from bytes to string
    output_text = output_text.decode('utf-8')

    # Split the output based on "Refining ligand" which indicates the start of a new ligand chunk
    ligand_chunks = re.split(r"\nRefining ligand", output_text)
    
    # The first chunk is often empty or irrelevant, so we remove it
    if not ligand_chunks[0].strip():
        ligand_chunks.pop(0)
    
    # Prepend the "Refining ligand" to each chunk for consistency
    ligand_chunks = ["Refining ligand" + chunk for chunk in ligand_chunks]
    
    return ligand_chunks

def extract_energies_and_ids(ligand_chunks):
    """
    Extracts ligand IDs and free energy values from each ligand chunk.
    
    Parameters:
    ligand_chunks (list of str): Each element is a chunk of text for one ligand.
    
    Returns:
    list of dict: Each dictionary contains the ligand ID and a list of free energy values.
    """
    results = []
    
    for chunk in ligand_chunks:
        # Extract ligand ID (using regex to capture the filename part after './test_out/')
        ligand_id_match = re.search(r"Refining ligand \./test_out/([^ ]+)", chunk)
        if ligand_id_match:
            ligand_id = ligand_id_match.group(1)
        else:
            continue  # Skip if no ID is found (unexpected case)
        
        # Find all affinity values (free energies in kcal/mol) in the chunk
        affinities = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)", chunk, re.MULTILINE)
        affinities = [float(affinity) for affinity in affinities]
        
        # Store results in a dictionary format
        results.append({
            "ligand_id": ligand_id,
            "affinities": affinities
        })
    
    return results


def standardize_mol(mol):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    carbon = Chem.MolFromSmarts('[#6]')
    salts = Chem.MolFromSmarts('[Na,Zn]')
    try:
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        mol = chooser.choose(mol)
        mol = charger.uncharge(mol)
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        # remove SMILES that do not contain carbon
        if len(mol.GetSubstructMatches(carbon)) == 0:
            return None
        # remove SMILES that still contain salts
        if len(mol.GetSubstructMatches(salts)) > 0:
            return None
        return Chem.CanonSmiles(smileR)
    except:
        print('Parsing Error:', Chem.MolToSmiles(mol))

    return None


def check_smiles(smiles, frags=None):
    shape = (len(smiles), 1) if frags is None else (len(smiles), 2)
    valids = np.zeros(shape)
    for j, smile in enumerate(smiles):
        # 1. Check if SMILES can be parsed by rdkit
        try:
            mol = Chem.MolFromSmiles(smile)
            valids[j, 0] = 0 if mol is None else 1
        except:
            valids[j, 0] = 0
        if frags is not None:
            # 2. Check if SMILES contain given fragments
            try:
                subs = frags[j].split('.')
                subs = [Chem.MolFromSmiles(sub) for sub in subs]
                valids[j, 1] = np.all([mol.HasSubstructMatch(sub) for sub in subs])
            except:
                valids[j, 1] = 0
    return valids


def compress_string(string):
    """
    Compresses a string
    Arguments:

        string (str)              : string to compress  
    Returns:
        compressed (str)          : compressed string
    """ 
    return zlib.compress(string.encode('utf-8')).hex()


def decompress_string(compressed):
    """
    Decompresses a compressed string
    Arguments:
        compressed (str)          : compressed string
    Returns:
        string (str)              : decompressed string
    """
    return zlib.decompress(bytes.fromhex(compressed)).decode('utf-8')


def write_to_log(log_path, smiles, target, scores, pdbqt_path=None, **kwargs):
    """
    Writes a log file
    Arguments:
        log_path (str)            : path to log file
        smiles (str)              : SMILES of ligand
        target (str)              : target name
        scores (list)             : list of scores
        pdbqt_path (str)          : path to pdbqt file
    """
    kwargs = {k: str(v) for k, v in kwargs.items()}

    if not os.path.isfile(log_path):
        with open(os.path.join(log_path), 'w') as f:
            header = '\t'.join(['smiles', 'target', 'scores'] + list(kwargs.keys()) + ['pdbqt'])
            f.write(header + '\n')

    if pdbqt_path is not None:
        with open(pdbqt_path, 'r') as f:
            pdbqt = f.read()
        pdbqt = compress_string(pdbqt)
    else:
        pdbqt = ''
    
    if not isinstance(scores, list):
        scores = [scores]
    
    z = [str(score) for score in  scores]
    if len(z) == 1:
        scores = z[0]
    else:
        scores = ';'.join(z)

    with open(log_path, 'a') as f:
        
        f.write('\t'.join([smiles, target, scores] + list(kwargs.values()) + [pdbqt]) + '\n')
    

def read_log(log_path):
    """
    Reads a log file
    Arguments:
        log_path (str)            : path to log file
    Returns:
        log (list)                : list of log entries
    """
    log = []
    with open(log_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            smiles, target, scores, pdbqt = line.strip().split('\t')
            scores = [float(score) for score in scores.split(';')]
            pdbqt = decompress_string(pdbqt)
            log += [(smiles, target, scores, pdbqt)]
    return log
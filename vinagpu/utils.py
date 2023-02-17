from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np

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
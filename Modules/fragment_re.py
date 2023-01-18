import numpy as np
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import BRICS

from rdkit import Chem
from rdkit.Chem import rdmolops
def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]


def mols_from_smiles(mols):
    return [mol_from_smiles(m) for m in mols]


def mol_to_graph_data(mol):
    A = rdmolops.GetAdjacencyMatrix(mol)
    node_features, edge_features = {}, {}

    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]

    for idx in range(A.shape[0]):
        atomic_num = mol.GetAtomWithIdx(idx).GetAtomicNum()
        node_features[idx]["label"] = int(atomic_num)

    for b1, b2 in bondidxs:
        btype = mol.GetBondBetweenAtoms(b1, b2).GetBondTypeAsDouble()
        edge_features[(b1, b2)]["label"] = int(btype)

    return A, node_features, edge_features

dummy = Chem.MolFromSmiles('[*]')

def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res
def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms

def fragment_iterative(mol, min_length=3):

    bond_data = list(BRICS.FindBRICSBonds(mol))

    try:
        idxs, labs = zip(*bond_data)
    except Exception:
        return []

    bonds = []
    for a1, a2 in idxs:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bonds.append(bond.GetIdx())

    order = np.argsort(bonds).tolist()
    bonds = [bonds[i] for i in order]

    frags, temp = [], deepcopy(mol)
    for bond in bonds:
        res = break_on_bond(temp, bond)

        if len(res) == 1:
            frags.append(temp)
            break
        head, tail = res
        if get_size(head) < min_length or get_size(tail) < min_length:
            continue

        frags.append(head)
        temp = deepcopy(tail)

    return frags

def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if bonds == []:
            frags.append(mol)
            return frags

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]],
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        frags.append(head)

        return fragment_recursive(tail, frags)
    except Exception:
        pass



def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    Chem.Kekulize(joined)
    return joined


def has_dummy_atom(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return True
    return False


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count
def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    try:
        if count_dummies(frags[0]) != 1:
            return None, None

        if count_dummies(frags[-1]) != 1:
            return None, None

        for frag in frags[1:-1]:
            if count_dummies(frag) != 2:
                return None, None

        mol = join_molecules(frags[0], frags[1])
        for i, frag in enumerate(frags[2:]):
            mol = join_molecules(mol, frag)


        mol_to_smiles(mol)

        return mol, frags
    except Exception:
        return None, None




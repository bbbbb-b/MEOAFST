from mol_utils import drop_salt
from rdkit import Chem


def read_file(file_name, drop_first=True):
    
    molObjects = []
    sum=0
    with open(file_name) as f:
        for l in f:
            if drop_first:
                drop_first = False
                continue
            sum+=1
            l = l.strip().split(",")[0]
            smi = drop_salt(l.strip())
            molObjects.append(Chem.MolFromSmiles(smi))
    return molObjects


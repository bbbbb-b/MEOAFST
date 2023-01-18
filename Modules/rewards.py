import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
from build_encoding import decode
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
from rdkit.Chem import Descriptors
from property_ecmop import qed,plogP,drd2,jnk,gsk


evaluated_mols = {}




def modify_fragment(f, swap):
    f[-(1+swap)] = (f[-(1+swap)] + 1) % 2
    return f





def get_key(fs):
    return tuple([np.sum([(int(x)* 2 ** (len(a) - y))
                    for x,y in zip(a, range(len(a)))]) if a[0] == 1 \
                     else 0 for a in fs])





def evaluate_chem_mol(mol):
    try:
        Chem.GetSSSR(mol)
        clogp = Crippen.MolLogP(mol)
        mw = MolDescriptors.CalcExactMolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        ret_val = [
            True,
            320 < mw < 420,
            2 < clogp < 3,
            40 < tpsa < 60
        ]
    except:
        ret_val = [False] * 4

    return ret_val
def grade_evaluate_chem_mol(mol):
    try:
        ret_val=[True]
        Chem.GetSSSR(mol)
        clogp = Crippen.MolLogP(mol)
        mw = MolDescriptors.CalcExactMolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        ret_val.append(round(mw, 3))
        ret_val.append(round(clogp, 3))
        ret_val.append(round(tpsa, 3))
    except:
        ret_val = [False] * 4

    return ret_val

def grade_evaluate_chem_mol_plogP(mol):
    try:
        ret_val=[True]
        plogP_=plogP(Chem.MolToSmiles(mol))
        ret_val.append(round(plogP_, 3))
    except:
        ret_val = [False] * 2

    return ret_val


def grade_evaluate_chem_mol_qed(mol):
    try:
        ret_val=[True]
        qed_=qed(Chem.MolToSmiles(mol))
        ret_val.append(round(qed_, 3))
    except:
        ret_val = [False] * 2

    return ret_val



def grade_evaluate_chem_mol_task1(mol):
    try:
        ret_val=[True]
        qed_ = qed(Chem.MolToSmiles(mol))
        plogP_=plogP(Chem.MolToSmiles(mol))
        ret_val.append(round(qed_,3))
        ret_val.append(round(plogP_, 3))
    except:
        ret_val = [False] * 3

    return ret_val

def grade_evaluate_chem_mol_task2(mol):
    try:
        ret_val=[True]
        qed_ = qed(Chem.MolToSmiles(mol))
        ddr2_=drd2(Chem.MolToSmiles(mol))
        ret_val.append(round(qed_,3))
        ret_val.append(round(ddr2_, 3))
    except:
        ret_val = [False] * 3

    return ret_val

def evaluate_mol(fs, epoch, decodings):

    global evaluated_mols

    key = get_key(fs)

    if key in evaluated_mols:
        return evaluated_mols[key][0]

    try:
        # mol = decode(fs, decodings)
        mol,_ = decode(fs, decodings)
        ret_val = evaluate_chem_mol(mol)
    except:
        ret_val = [False] * 4

    evaluated_mols[key] = (np.array(ret_val), epoch)

    return np.array(ret_val)



def get_reward(fs,epoch,dist):

    if fs[fs[:,0] == 0].sum() < 0:
        return -0.1

    return (dist * evaluate_mol(fs, epoch)).sum()



def get_init_dist(X, decodings):

    arr = np.asarray([evaluate_mol(X[i], -1, decodings) for i in range(X.shape[0])])
    dist = arr.shape[0] / (1.0 + arr.sum(0))
    return dist


def clean_good(X, decodings):
    # X = [X[i] for i in range(X.shape[0]) if not
    #     evaluate_mol(X[i], -1, decodings).all()]
    # return np.asarray(X)
    X_=[]
    for i in X:
        from mol_utils import decode
        mol,_=decode(i,decodings)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if sum(evaluate_chem_mol(mol))==4:
            continue
        else:
            X_.append(i)
    return np.asarray(X_)


def clean_good_task(X, decodings,isfmpo):
    # X = [X[i] for i in range(X.shape[0]) if not
    #     evaluate_mol(X[i], -1, decodings).all()]
    # return np.asarray(X)
    X_=[]
    for i in range(X.shape[0]):
        from mol_utils import decode
        if isfmpo:
            mol=decode(X[i],decodings,isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)
        if not mol or mol is None:
            continue
        else:
            X_.append(X[i])
    return np.asarray(X_)

def morgan_fingerprint(mol):

    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(smi1, smi2):

    fp1 = morgan_fingerprint(Chem.MolFromSmiles(smi1))
    fp2 = morgan_fingerprint(Chem.MolFromSmiles(smi2))
    return rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)


def judge(smi1,smi2):
    if (qed(smi1)==qed(smi2) and drd2(smi1)==drd2(smi2) and plogP(smi1)==plogP(smi2)) or tanimoto_similarity(smi1,smi2)==1:
        return True
    else:
        return False

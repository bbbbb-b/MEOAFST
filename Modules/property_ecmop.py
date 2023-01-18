from tdc import Oracle
from rdkit.Chem import Crippen
from rdkit import Chem

qed_ = Oracle('qed')
sa_ = Oracle('sa')
drd2_ = Oracle('drd2')
jnk_ = Oracle('JNK3')
gsk_ = Oracle('GSK3B')

def qed(smi):
    return qed_(smi)


def plogP(smi):
    return Crippen.MolLogP(Chem.MolFromSmiles(smi))-sa_(smi)

def drd2(smi):
    return drd2_(smi)

def jnk(smi):
    return jnk_(smi)

def gsk(smi):
    return gsk_(smi)


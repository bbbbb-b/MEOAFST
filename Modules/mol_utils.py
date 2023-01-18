from global_parameters import MOL_SPLIT_START, MAX_FREE, MAX_ATOMS, MAX_FRAGMENTS,MIN_FRAGMENTS
from rdkit import Chem
import numpy as np
import random
from build_encoding import decode,decode_notree
from  rewards import grade_evaluate_chem_mol,grade_evaluate_chem_mol_plogP,grade_evaluate_chem_mol_qed,grade_evaluate_chem_mol_task1,grade_evaluate_chem_mol_task2
from fragment_re import fragment_recursive,reconstruct
from rdkit.Chem import AllChem
import rdkit

NOBLE_GASES = set([2, 10, 18, 36, 54, 86])
ng_correction = set()


def drop_salt(s):
    s = s.split(".")
    return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]




def okToBreak(bond):

    if bond.IsInRing():
        return False

    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False


    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    if not(begin_atom.IsInRing() or end_atom.IsInRing()):
        return False
    elif begin_atom.GetAtomicNum() >= MOL_SPLIT_START or \
            end_atom.GetAtomicNum() >= MOL_SPLIT_START:
        return False
    else:
        return True

def split_molecule(mol):

    split_id = MOL_SPLIT_START  ## 70

    res = []
    to_check = [mol]
    while len(to_check) > 0:
        ms = spf(to_check.pop(), split_id)
        if len(ms) == 1:
            res += ms
        else:
            to_check += ms
            split_id += 1

    return create_chain(res)


def spf(mol, split_id):

    bonds = mol.GetBonds()
    for i in range(len(bonds)):
        if okToBreak(bonds[i]):
            mol = Chem.FragmentOnBonds(mol, [i], addDummies=True, dummyLabels=[(0, 0)])

            n_at = mol.GetNumAtoms()
            mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
            mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
            return Chem.rdmolops.GetMolFrags(mol, asMols=True)

    return [mol]



def create_chain(splits):
    splits_ids = np.asarray(
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits])

    splits_ids = \
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits]

    splits2 = []
    mv = np.max(splits_ids)
    look_for = [mv if isinstance(mv, np.int64) else mv[0]]
    join_order = []

    mols = []

    for i in range(len(splits_ids)):
        l = splits_ids[i]
        if l[0] == look_for[0] and len(l) == 1:
            mols.append(splits[i])
            splits2.append(splits_ids[i])
            splits_ids[i] = []


    while len(look_for) > 0:
        sid = look_for.pop()
        join_order.append(sid)
        next_mol = [i for i in range(len(splits_ids))
                      if sid in splits_ids[i]]

        if len(next_mol) == 0:
            break
        next_mol = next_mol[0]

        for n in splits_ids[next_mol]:
            if n != sid:
                look_for.append(n)
        mols.append(splits[next_mol])
        splits2.append(splits_ids[next_mol])
        splits_ids[next_mol] = []

    return [simplify_splits(mols[i], splits2[i], join_order) for i in range(len(mols))]




def simplify_splits(mol, splits, join_order):

    td = {}
    n = 0
    for i in splits:
        for j in join_order:
            if i == j:
                td[i] = MOL_SPLIT_START + n
                n += 1
                if n in NOBLE_GASES:
                    n += 1


    for a in mol.GetAtoms():
        k = a.GetAtomicNum()
        if k in td:
            a.SetAtomicNum(td[k])

    return mol



def get_join_list(mol):

    join = []
    rem = []
    bonds = []

    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        if an >= MOL_SPLIT_START:
            while len(join) <= (an - MOL_SPLIT_START):
                rem.append(None)
                bonds.append(None)
                join.append(None)

            b = a.GetBonds()[0]
            ja = b.GetBeginAtom() if b.GetBeginAtom().GetAtomicNum() < MOL_SPLIT_START else \
                 b.GetEndAtom()
            join[an - MOL_SPLIT_START] = ja.GetIdx()
            rem[an - MOL_SPLIT_START] = a.GetIdx()
            bonds[an - MOL_SPLIT_START] = b.GetBondType()
            a.SetAtomicNum(0)

    return [x for x in join if x is not None],\
           [x for x in bonds if x is not None],\
           [x for x in rem if x is not None]



def join_fragments(fragments):

    to_join = []
    bonds = []
    pairs = []
    del_atoms = []
    new_mol = fragments[0]
    j,b,r = get_join_list(fragments[0])
    to_join += j
    del_atoms += r
    bonds += b
    offset = fragments[0].GetNumAtoms()

    for f in fragments[1:]:
        if len(to_join)==0:
            return False
        j,b,r = get_join_list(f)
        p = to_join.pop()
        pb = bonds.pop()

        # Check bond types if b[:-1] == pb
        if b[:-1] != pb:
            assert("Can't connect bonds")



        pairs.append((p, j[-1] + offset,pb))

        for x in j[:-1]:
            to_join.append(x + offset)
        for x in r:
            del_atoms.append(x + offset)
        bonds += b[:-1]

        offset += f.GetNumAtoms()
        new_mol = Chem.CombineMols(new_mol, f)


    new_mol =  Chem.EditableMol(new_mol)

    for a1,a2,b in pairs:
        new_mol.AddBond(a1,a2, order=b)

    # Remove atom with greatest number first:
    for s in sorted(del_atoms, reverse=True):
        new_mol.RemoveAtom(s)
    return new_mol.GetMol()





def get_class(fragment):

    is_ring = False
    n = 0

    for a in fragment.GetAtoms():
        if a.IsInRing():
            is_ring = True

        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1

    smi = Chem.MolToSmiles(fragment)

    if n == 1:
        cl = "R-group"
    elif is_ring:
        cl = "Scaffold-" + str(n)
    else:
        cl = "Linker-" + str(n)

    return cl





def should_use(fragment):

    n = 0
    m = 0
    for a in fragment.GetAtoms():
        m += 1
        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1
        if n > MAX_FREE or m > MAX_ATOMS:
            return False

    return True




def get_fragments(mols,isfmpo):

    used_mols = np.zeros(len(mols)) != 0

    fragments = dict()

    i = -1
    for mol in mols:
        i += 1
        try:
            if isfmpo:
                fs = split_molecule(mol)
            else:
                fs=fragment_recursive(mol,[])
        except:
            continue

        if len(fs) <= MAX_FRAGMENTS and all(map(should_use, fs)) and len(fs)>=MIN_FRAGMENTS:
            used_mols[i] = True
        else:
            continue

        for f in fs:
            cl = get_class(f)
            fragments[Chem.MolToSmiles(f)] = (f, cl)
    return fragments, used_mols

def mol_ec(mol,p,ec_epoch,tree_sun=2):
    all_pop=[]   ## 变异ec_epoch个分子
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in mol if e[0] == 1]   ## 有效的碎片
    n_mol = len(enc)

    while len(all_pop)<ec_epoch:
        from copy import deepcopy
        all_pop.append(deepcopy(mol))
        all_pop=np.asarray(all_pop)
        pos = np.random.randint(0, n_mol, 1)[0]
        frag_pos = np.random.randint(0, tree_sun, 1)[0]
        location = find_n(len(enc[0]), p)
        all_pop[len(all_pop)-1][pos][location] = frag_pos
        all_pop = np.unique(all_pop,axis=0)
        all_pop=all_pop.tolist()
    all_pop.append(mol)

    return np.asarray(all_pop)



def mol_ec_random_tree_notree(mol,p,ec_epoch,tree_sun,random):
    all_pop=[]
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in mol if e[0] == 1]
    n_mol = len(enc)

    while len(all_pop)<ec_epoch:
        from copy import deepcopy
        all_pop.append(deepcopy(mol))
        all_pop=np.asarray(all_pop)
        pos = np.random.randint(0, n_mol, 1)[0]
        r=np.random.random()


        if r>random:
            frag_pos = np.random.randint(0, tree_sun, 1)[0]
            location = find_n(len(enc[0]), p)
            all_pop[len(all_pop)-1][pos][location] = frag_pos
        else:
            i=1
            while i<=len(enc[0]):
                frag_pos = np.random.randint(0, tree_sun, 1)[0]
                all_pop[len(all_pop) - 1][pos][i] = frag_pos
                i+=1

        all_pop = np.unique(all_pop,axis=0)
        all_pop=all_pop.tolist()
    all_pop.append(mol)

    return np.asarray(all_pop)



def mol_ec_random_of_nIter_tree_notree(mol,p,ec_epoch,tree_sun,iter,nIter):
    all_pop=[]   ## 变异ec_epoch个分子
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in mol if e[0] == 1]
    n_mol = len(enc)

    while len(all_pop)<ec_epoch:
        from copy import deepcopy
        all_pop.append(deepcopy(mol))
        all_pop=np.asarray(all_pop)
        pos = np.random.randint(0, n_mol, 1)[0]
        r=np.random.random()
        import math

        if r > (np.cos(math.pi / 2 * (iter / nIter))):

            frag_pos = np.random.randint(0, tree_sun, 1)[0]
            location = find_n(len(enc[0]), p)
            all_pop[len(all_pop)-1][pos][location] = frag_pos
        else:
            # 随机变异
            i=1
            while i<=len(enc[0]):
                frag_pos = np.random.randint(0, tree_sun, 1)[0]
                all_pop[len(all_pop) - 1][pos][i] = frag_pos
                i+=1

        all_pop = np.unique(all_pop,axis=0)
        all_pop=all_pop.tolist()
    all_pop.append(mol)

    return np.asarray(all_pop)


def mol_ec_notree(mol,p,ec_epoch,tree_sun,len_fragment):
    all_pop=[]
    enc=[str(int(e)) for e in mol if e!=-1]
    n_mol = len(enc)
    while len(all_pop)<ec_epoch:
        from copy import deepcopy
        all_pop.append(deepcopy(mol))
        all_pop=np.asarray(all_pop)
        pos = np.random.randint(0, n_mol, 1)[0]
        frag_pos = np.random.randint(0, len_fragment, 1)[0]

        all_pop[len(all_pop)-1][pos] = str(int(frag_pos))
        all_pop = np.unique(all_pop,axis=0)
        all_pop=all_pop.tolist()
    all_pop.append(mol)

    return np.asarray(all_pop)


def child_mol_ec(pops,p,all_ec_epoch, ec_epoch, tree_sum):
    all_pop = []
    for mol in pops:
        all_pop=np.asarray(all_pop).reshape((-1,mol.shape[0],mol.shape[1]))
        all_pop=np.vstack((all_pop,mol_ec(mol,p,ec_epoch,tree_sum)))
    try:
        all_pop = np.unique(all_pop, axis=0)
    except:
        print("ssss")

    while len(all_pop)<all_ec_epoch:
        all_pop = np.vstack((all_pop, child_mol_ec(pops,p,all_ec_epoch-len(all_pop),ec_epoch,tree_sum)))
        all_pop = np.unique(all_pop, axis=0)
    return np.asarray(all_pop)


def child_mol_ec_random_tree_notree(pops,p,all_ec_epoch, ec_epoch, tree_sum,random):
    all_pop = []
    for mol in pops:
        all_pop=np.asarray(all_pop).reshape((-1,mol.shape[0],mol.shape[1]))
        all_pop=np.vstack((all_pop,mol_ec_random_tree_notree(mol,p,ec_epoch,tree_sum,random)))
    try:
        all_pop = np.unique(all_pop, axis=0)
    except:
        print("ssss")

    while len(all_pop)<all_ec_epoch:
        all_pop = np.vstack((all_pop, child_mol_ec_random_tree_notree(pops,p,all_ec_epoch-len(all_pop),ec_epoch,tree_sum,random)))
        all_pop = np.unique(all_pop, axis=0)
    return np.asarray(all_pop)


def child_mol_ec_random_of_nIter_tree_notree(pops,p,all_ec_epoch, ec_epoch, tree_sum,iter,nIter):
    all_pop = []
    for mol in pops:
        all_pop=np.asarray(all_pop).reshape((-1,mol.shape[0],mol.shape[1]))
        all_pop=np.vstack((all_pop,mol_ec_random_of_nIter_tree_notree(mol,p,ec_epoch,tree_sum,iter,nIter)))
    try:
        all_pop = np.unique(all_pop, axis=0)
    except:
        print("ssss")

    while len(all_pop)<all_ec_epoch:
        all_pop = np.vstack((all_pop, child_mol_ec_random_of_nIter_tree_notree(pops,p,all_ec_epoch-len(all_pop),ec_epoch,tree_sum,iter,nIter)))
        all_pop = np.unique(all_pop, axis=0)
    return np.asarray(all_pop)


def child_mol_ec_notree(pops,p,all_ec_epoch, ec_epoch, tree_sum,len_fragment):
    all_pop = []
    for mol in pops:
        all_pop=np.asarray(all_pop).reshape((-1,mol.shape[0]))
        all_pop=np.vstack((all_pop,mol_ec_notree(mol,p,ec_epoch,tree_sum,len_fragment)))
    try:
        all_pop = np.unique(all_pop, axis=0)
    except:
        print("ssss")

    while len(all_pop)<all_ec_epoch:
        all_pop = np.vstack((all_pop, child_mol_ec_notree(pops,p,all_ec_epoch-len(all_pop),ec_epoch,tree_sum,len_fragment)))
        all_pop = np.unique(all_pop, axis=0)
    return np.asarray(all_pop)


def find_n(sum,p):
    data = {}
    pick_value = -1
    value_sum = 0
    q=1
    for i in range(sum):
        data[i] = p + i*q
        value_sum += p +i*q
    t = random.randint(0, value_sum - 1)
    for key, value in data.items():
        t -= value
        if t < 0:
            pick_value = key
            break
    return pick_value+1


# 得分
def mol_grade(X,decodings):
    fits=[]
    for i in range(X.shape[0]):
        mol,_= decode(X[i], decodings)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            ret_val = grade_evaluate_chem_mol(mol)
        fits.append(ret_val)
    return np.asarray(fits)

# 得分plogP
def mol_grade_plogP(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                if sim < 0.6:
                    ret_val = [False, False]
                else:
                    ret_val = grade_evaluate_chem_mol_plogP(mol)
            except:
                ret_val = [False, False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_qed(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings, isfmpo)

        if not mol or mol is None:
            ret_val = [False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                sim1=tanimoto_similarity(Chem.MolToSmiles(mol1),Chem.MolToSmiles(pre_mol))
                if sim < 0.4:
                    ret_val = [False, False]
                else:
                    ret_val = grade_evaluate_chem_mol_qed(mol)
            except:
                ret_val = [False, False]
        fits.append(ret_val)
    return np.asarray(fits)



def mol_grade_task1(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                if sim < 0.3:
                    ret_val = [False, False,False]
                else:
                    ret_val = grade_evaluate_chem_mol_task1(mol)
            except:
                ret_val = [False, False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_task1_sim(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                ret_val = grade_evaluate_chem_mol_task1(mol)
                ret_val.append(sim)
            except:
                ret_val = [False,False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_task1_sim_notree(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode_notree(X[i], decodings, isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                ret_val = grade_evaluate_chem_mol_task1(mol)
                ret_val.append(round(sim,5))
            except:
                ret_val = [False,False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)



def mol_grade_task1_or_sim(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                if sim < 0.3:
                    ret_val = [False, False,False,False]
                else:
                    ret_val = grade_evaluate_chem_mol_task1(mol)
                    ret_val.append(sim)
            except:
                ret_val = [False,False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)



def mol_grade_task2(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                if sim < 0.3:
                    ret_val = [False, False,False]
                else:
                    ret_val = grade_evaluate_chem_mol_task2(mol)
            except:
                ret_val = [False, False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_task2_sim(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                ret_val = grade_evaluate_chem_mol_task2(mol)
                ret_val.append(sim)
            except:
                ret_val = [False, False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_task2_sim_notree(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode_notree(X[i], decodings, isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                ret_val = grade_evaluate_chem_mol_task2(mol)
                ret_val.append(round(sim,5))
            except:
                ret_val = [False,False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def mol_grade_task2_or_sim(X,pre_mol,decodings,isfmpo):
    fits=[]
    for i in range(X.shape[0]):
        if isfmpo:
            mol = decode(X[i], decodings, isfmpo)
        else:
            mol, _ = decode(X[i], decodings,isfmpo)

        if not mol or mol is None:
            ret_val = [False,False,False,False]
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            pre_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(pre_mol))
            try:
                sim = tanimoto_similarity(Chem.MolToSmiles(mol1), Chem.MolToSmiles(pre_mol1))
                if sim < 0.3:
                    ret_val = [False, False, False,False]
                else:
                    ret_val = grade_evaluate_chem_mol_task2(mol)
                    ret_val.append(sim)
            except:
                ret_val = [False, False,False,False]
        fits.append(ret_val)
    return np.asarray(fits)


def pre_mol_grade(X,decodings):
    mol,_ = decode(X, decodings)
    ret_val = grade_evaluate_chem_mol(mol)
    return np.asarray(ret_val)


def pre_mol_grade_plogP(mol):
    ret_val = grade_evaluate_chem_mol_plogP(mol)
    return np.asarray(ret_val)



def pre_mol_grade_qed(mol):
    ret_val = grade_evaluate_chem_mol_qed(mol)
    return np.asarray(ret_val)


def pre_mol_grade_task1(mol):

    ret_val = grade_evaluate_chem_mol_task1(mol)
    return np.asarray(ret_val)


def pre_mol_grade_task2(mol):

    ret_val = grade_evaluate_chem_mol_task2(mol)
    return np.asarray(ret_val)


def select_pop(all_pop,fits,num):
    all_fits=[]
    for fit in fits:
        all_fits.append(all_grade(fit))
    all_fits = np.array(all_fits)
    all_pop=np.array(all_pop)
    fits=np.array(fits)
    arrIndex = np.array(all_fits).argsort()[::-1]
    return all_pop[arrIndex][:num],fits[arrIndex][:num],all_fits[arrIndex][:num]


def select_pop_plogP(all_pop,fits,num):
    all_fits=[]
    for fit in fits:
        all_fits.append(all_grade_plogP(fit))
    all_fits = np.array(all_fits)
    all_pop=np.array(all_pop)
    fits=np.array(fits)
    arrIndex = np.array(all_fits).argsort()[::-1]
    return all_pop[arrIndex][:num],fits[arrIndex][:num],all_fits[arrIndex][:num]


def select_pop_qed(all_pop,fits,num):
    all_fits=[]
    for fit in fits:
        all_fits.append(all_grade_qed(fit))
    all_fits = np.array(all_fits)
    all_pop=np.array(all_pop)
    fits=np.array(fits)
    arrIndex = np.array(all_fits).argsort()[::-1]
    return all_pop[arrIndex][:num],fits[arrIndex][:num],all_fits[arrIndex][:num]


def select_pop_task1(pool, pops, fits,pre_fits, ranks, distances):

    nPop = pops.shape[0]
    index_judge=np.zeros(nPop)
    nF = fits.shape[1]-1
    newPops = []
    newFits = []
    indices = np.arange(nPop).tolist()
    n=np.sum(ranks==0)
    index=np.arange(nPop)
    i = 0
    if n>pool:
        rIdices = index[ranks==0]
        for idx in rIdices:
            if fits[idx][1]>=0.85 and fits[idx][2]-pre_fits[2]>=3 and fits[idx][3]>=0.3:
                newPops.append(pops[idx])
                newFits.append(fits[idx])
                index_judge[idx]=1
                i+=1
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)
        idx = compare(idx1, idx2, ranks, distances)
        if index_judge[idx]!=1:
            index_judge[idx]=1
            newPops.append(pops[idx])
            newFits.append(fits[idx])
            i += 1
    return np.asarray(newPops), np.asarray(newFits)


def select_pop_task2(pool, pops, fits, ranks, distances):


    nPop = pops.shape[0]
    index_judge=np.zeros(nPop)
    nF = fits.shape[1]-1
    newPops = []
    newFits = []
    indices = np.arange(nPop).tolist()
    n=np.sum(ranks==0)
    index=np.arange(nPop)
    i = 0
    if n>pool:
        rIdices = index[ranks==0]
        for idx in rIdices:
            if fits[idx][1]>=0.8 and fits[idx][2]>=0.4 and fits[idx][3]>=0.3:
                newPops.append(pops[idx])
                newFits.append(fits[idx])
                index_judge[idx]=1
                i+=1

    while i < pool:
        idx1, idx2 = random.sample(indices, 2)
        idx = compare(idx1, idx2, ranks, distances)
        if index_judge[idx]!=1:
            index_judge[idx]=1
            newPops.append(pops[idx])
            newFits.append(fits[idx])
            i += 1
    return np.asarray(newPops), np.asarray(newFits)


def compare(idx1, idx2, ranks, distances):

    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx


def all_grade_plogP(fits):
    if fits[0]:
        totle=fits[1]
    else:
        totle=-10000
    return round(totle, 3)



def all_grade_qed(fits):
    if fits[0]:
        totle=fits[1]
    else:
        totle=-10000
    return round(totle, 3)




def all_grade(fits):
    min_mv=320
    max_mv=420
    min_clogp=2
    max_clogp=3
    min_tpsa=40
    max_tpsa=60
    if fits[0]==True:
        totle=0
        if fits[1] < min_mv:
            totle -= (min_mv - fits[1]) * (min_clogp / min_mv)
        elif fits[1] > max_mv:
            totle -= (fits[1] - max_mv) * (max_clogp / max_mv)
        if fits[2] < min_clogp:
            totle -= (min_clogp - fits[2])
        elif fits[2] > max_clogp:
            totle -= (fits[2] - max_clogp)
        if fits[3] < min_tpsa:
            totle -= (min_tpsa - fits[3]) * (min_tpsa / min_mv)
        elif fits[3] > max_tpsa:
            totle -= (fits[3] - max_tpsa) * (max_tpsa / max_mv)
    else:
        totle=-10000
    return round(totle, 3)



def morgan_fingerprint(mol):

    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(smi1, smi2):

    mol1=Chem.MolFromSmiles(smi1)
    mol2=Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0
    fp1 = morgan_fingerprint(mol1)
    fp2 = morgan_fingerprint(mol2)

    return rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)

def nonDominationSort(pops, fits):

    for fit in fits:
        if not fit[0]:
            fit[1]=-10000
            fit[2]=-10000
            fit[3] = -10000
    nPop = pops.shape[0]
    nF = fits.shape[1]-1
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)
    sPs = []
    for i in range(nPop):
        iSet = []
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i][1:] >= fits[j][1:]
            isDom2 = fits[i][1:] > fits[j][1:]

            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)

            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)
    r = 0
    indices = np.arange(nPop)
    while sum(nPs==0) != 0:
        rIdices = indices[nPs==0]
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1
        r += 1
    return ranks


def crowdingDistanceSort(pops, fits, ranks):

    for fit in fits:
        if not fit[0]:
            fit[1]=-10000
            fit[2]=-10000
            fit[3] = -10000
    nPop = pops.shape[0]
    nF = fits.shape[1]-1
    dis = np.zeros(nPop)
    nR = ranks.max()
    indices = np.arange(nPop)
    for r in range(nR+1):
        rIdices = indices[ranks==r]
        rPops = pops[ranks==r]
        rFits = fits[ranks==r][:,1:]
        rSortIdices = np.argsort(rFits, axis=0)
        rSortFits = np.sort(rFits,axis=0)
        fMax = np.max(rFits,axis=0)
        fMin = np.min(rFits,axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:,i]]
            j = 1
            while n > 2 and j < n-1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf
    return dis


def optSelect_task1(num,MergePops, MergeFits,pre_fits):

    newPops = []
    newFits = []
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    n = np.sum(MergeRanks == 0)
    index=np.arange(MergePops.shape[0])
    index_judge=np.zeros(MergePops.shape[0],dtype=np.int32)
    if n>num:
        rIdices = index[MergeRanks==0]
        for idx in rIdices:
            if MergeFits[idx][1] >= 0.85 and MergeFits[idx][2] - pre_fits[2] >= 3 and MergeFits[idx][3] >= 0.3:
                newPops.append(MergePops[idx])
                newFits.append(MergeFits[idx])
                index_judge[idx] = 1
                i += 1
    rIndices1 = indices[MergeRanks == r]
    rIndices2 = indices[index_judge!=1]
    rIndices=list(set(rIndices1).intersection(set(rIndices2)))
    rIndices=np.asarray(rIndices)
    while i + len(rIndices) <= num:
        for rI in rIndices:
            newPops.append(MergePops[rI])
            newFits.append(MergeFits[rI])
        r += 1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]

    if i < num:
        rDistances = MergeDistances[rIndices]
        rSortedIdx = np.argsort(rDistances)[::-1]
        surIndices = rIndices[rSortedIdx[:(num - i)]]
        newPops.extend(MergePops[surIndices])
        newFits.extend(MergeFits[surIndices])
    return np.asarray(newPops),np.asarray(newFits)


def optSelect_task2(num,MergePops, MergeFits):

    newPops = []
    newFits = []
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    n = np.sum(MergeRanks == 0)
    index=np.arange(MergePops.shape[0])
    index_judge=np.zeros(MergePops.shape[0])
    if n>num:
        rIdices = index[MergeRanks==0]
        for idx in rIdices:
            if MergeFits[idx][1] >= 0.8 and MergeFits[idx][2] >= 0.4 and MergeFits[idx][3] >= 0.3:
                newPops.append(MergePops[idx])
                newFits.append(MergeFits[idx])
                index_judge[idx] = 1
                i += 1
    rIndices1 = indices[MergeRanks == r]
    rIndices2 = indices[index_judge != 1]
    rIndices = list(set(rIndices1).intersection(set(rIndices2)))
    rIndices=np.asarray(rIndices)
    while i + len(rIndices) <= num:
        for rI in rIndices:
            newPops.append(MergePops[rI])
            newFits.append(MergeFits[rI])
        r += 1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]

    if i < num:
        rDistances = MergeDistances[rIndices]
        rSortedIdx = np.argsort(rDistances)[::-1]
        surIndices = rIndices[rSortedIdx[:(num - i)]]
        newPops.extend(MergePops[surIndices])
        newFits.extend(MergeFits[surIndices])
    return np.asarray(newPops),np.asarray(newFits)

def optSelect(MergePops, MergeFits):

    newPops = []
    newFits = []
    MergeRanks = nonDominationSort(MergePops, MergeFits)

    indices = np.arange(MergePops.shape[0])
    r = 0
    rIndices = indices[MergeRanks == r]
    for rI in rIndices:
        newPops.append(MergePops[rI])
        newFits.append(MergeFits[rI])
    return np.asarray(newPops),np.asarray(newFits)


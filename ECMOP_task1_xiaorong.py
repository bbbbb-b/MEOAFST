import sys

sys.path.insert(0, './Modules/')
import pandas as pd
import numpy as np

from file_reader import read_file
from mol_utils import get_fragments, mol_ec_random_tree_notree, mol_grade, pre_mol_grade, select_pop, child_mol_ec_random_tree_notree, pre_mol_grade_plogP, \
    select_pop_plogP, mol_grade_plogP, tanimoto_similarity,pre_mol_grade_task1,mol_grade_task1_or_sim,mol_grade_task1_sim,select_pop_task1,nonDominationSort,crowdingDistanceSort,optSelect_task1,optSelect
from build_encoding import get_encodings, encode_molecule, decode_molecule, encode_list, save_decodings, \
    save_decodings_task1, save_encodings_task1
from rdkit import rdBase
import logging
from rdkit import Chem
import multiprocessing

logging.getLogger().setLevel(logging.INFO)
rdBase.DisableLog('rdApp.error')






def main(fragment_file, lead_file):
    isfmpo=True
    random=0
    p = 13
    ec_epoch = 29
    child_ec_epoch = 1
    tree_sum = 2
    nIter = 50
    chongqi = 0
    num = 15
    successs = multiprocessing.Queue()
    results = multiprocessing.Queue()
    result = []
    success = 0
    fragment_mols = []
    # fragment_mols = read_file(fragment_file)
    lead_mols = read_file(lead_file)
    fragment_mols += lead_mols
    logging.info("Read %s molecules for fragmentation library", len(fragment_mols))
    logging.info("Read %s lead moleculs", len(lead_mols))
    fragments, used_mols = get_fragments(fragment_mols,isfmpo)
    logging.info("Num fragments: %s", len(fragments))
    logging.info("Total molecules used: %s", len(used_mols))
    lead_mols = np.asarray(fragment_mols[-len(lead_mols):])[used_mols[-len(lead_mols):]]
    logging.info("Total lead molecules used: %s", len(lead_mols))
    assert len(fragments)
    assert len(used_mols)
    encodings, decodings = get_encodings(fragments,True,'History/task1/decodings_task1_qedplogp_0.85_0.txt','History/task1/encodings_task1_qedplogp_0.85_0.txt')
    # save_decodings_task1(decodings,'History/task1/decodings_task1_qedplogp_0.85_0.txt')
    # save_encodings_task1(encodings, 'History/task1/encodings_task1_qedplogp_0.85_0.txt')
    logging.info("Saved decodings")
    logging.info("Saved encodings")


    X = encode_list(lead_mols, encodings,isfmpo)
    logging.info("Building models")
    # from rewards import clean_good_task
    # X = clean_good_task(X, decodings,isfmpo)
    logging.info(f'X.shape: {X.shape}')
    # X = clean_good(X, decodings)
    locks = []
    for i in range(12):
        lock = multiprocessing.Lock()
        lock.acquire()
        locks.append(lock)

    p1 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 1, 12, locks[0],isfmpo,chongqi,random))
    p2 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 2, 12, locks[1],isfmpo,chongqi,random))
    p3 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 3, 12, locks[2],isfmpo,chongqi,random))
    p4 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 4, 12, locks[3],isfmpo,chongqi,random))
    p5 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 5, 12, locks[4],isfmpo,chongqi,random))
    p6 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 6, 12, locks[5],isfmpo,chongqi,random))
    p7 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 7, 12, locks[6],isfmpo,chongqi,random))
    p8 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 8, 12, locks[7],isfmpo,chongqi,random))
    p9 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 9, 12, locks[8],isfmpo,chongqi,random))
    p10 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 10, 12, locks[9],isfmpo,chongqi,random))
    p11 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 11, 12, locks[10],isfmpo,chongqi,random))
    p12 = multiprocessing.Process(target=ec, args=(
        lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, 12, 12, locks[11],isfmpo,chongqi,random))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    for lock in locks:
        with lock:
            pass

    while not successs.empty():
        success += successs.get()
    while not results.empty():
        result.extend(results.get())
    print(success)
    df_to_save_A_to_B = pd.DataFrame(result,
                                     columns=['smile','iter_sum'],index=None)
    df_to_save_A_to_B.to_csv('./resultData_xiaorong/p1~3_ep30_nIter50_cq0_3_0.85_sim0.3_random0.csv', index=False)

    logging.info("Training")
    # history = train(X, actor, critic, decodings)
    logging.info("Saving")
    # np.save("History/history.npy", history)


def ec(lead_mols,X, decodings, p, ec_epoch, tree_sum, num, nIter, child_ec_epoch, successs, results, index, total, lock,isfmpo,chongqi,random):
    result = []
    resplash = np.zeros(X.shape[0])
    success = 0
    aaa = int(X.shape[0] / total)
    start=aaa * (index - 1)
    end= aaa * index
    if index==12:
        end=X.shape[0]
    i=start

    Iter=[]
    print(f'=================进程{index}=================')


    while i<end:
        iter_sum = 0
        if resplash[i]==0:
            Iter=[]
        judge = False

        pre_fits = pre_mol_grade_task1(lead_mols[i])

        pre_all_pop = mol_ec_random_tree_notree(X[i], p, ec_epoch, tree_sum,random)

        pre_all_fits = mol_grade_task1_sim(pre_all_pop, lead_mols[i], decodings,isfmpo)

        ranks = nonDominationSort(pre_all_pop, pre_all_fits)
        distances = crowdingDistanceSort(pre_all_pop, pre_all_fits, ranks)
        pops, fits= select_pop_task1(num,pre_all_pop, pre_all_fits,pre_fits, ranks,distances)

        for iter in range(nIter):
            iter_sum+=1
            pops = child_mol_ec_random_tree_notree(pops, p, ec_epoch + 1, child_ec_epoch, tree_sum,random)
            # 得分
            fits = mol_grade_task1_sim(pops, lead_mols[i], decodings,isfmpo)
            # 选择
            pops, fits = optSelect_task1(num,pops, fits,pre_fits)

            for fit in fits:
                child_qed = fit[1]
                child_plogp = fit[2]
                child_sim = fit[3]
                if child_plogp - pre_fits[2] >= 3 and child_qed>=0.85 and child_sim >=0.3:
                    judge = True
                    break
                else:
                    judge = False
            if judge:
                break

        Iter.append(iter_sum)
        if  resplash[i]==chongqi:
            mol = lead_mols[i]
            pre_smi = Chem.MolToSmiles(mol)
            iter_sum_mean=np.mean(Iter)
            result.append([pre_smi,iter_sum_mean])
            print(f'第{i + 1}个分子iter_sum_mean:{iter_sum_mean}')
            i+=1
        else:
            resplash[i] += 1
            print(f'第{i + 1}个分子的第{resplash[i]}重启')

    successs.put(success)
    results.put(result)
    lock.release()


if __name__ == "__main__":

    fragment_file = "DataNew/task1/task1_qedplogp_0.85_0_fragment.csv"
    lead_file = "DataNew/task1/task1_qedplogp_test.csv"

    if len(sys.argv) > 1:
        fragment_file = sys.argv[1]

    if len(sys.argv) > 2:
        lead_file = sys.argv[2]

    main(fragment_file, lead_file)

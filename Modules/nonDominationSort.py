
import random 
import numpy as np

def nonDominationSort(pops, fits): 

    nPop = pops.shape[0] 
    nF = fits.shape[1]
    ranks = np.zeros(nPop, dtype=np.int32)  
    nPs = np.zeros(nPop)
    sPs = []
    for i in range(nPop): 
        iSet = []
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] >= fits[j]
            isDom2 = fits[i] > fits[j]

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

    nPop = pops.shape[0] 
    nF = fits.shape[1]
    dis = np.zeros(nPop) 
    nR = ranks.max()
    indices = np.arange(nPop) 
    for r in range(nR+1):
        rIdices = indices[ranks==r]
        rPops = pops[ranks==r]
        rFits = fits[ranks==r]
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



if __name__ == "__main__":
    y1 = np.arange(1,5).reshape(4,1)
    y2 = 5 - y1 
    fit1 = np.concatenate((y1,y2),axis=1) 
    y3 = 6 - y1 
    fit2 = np.concatenate((y1,y3),axis=1)
    y4 = 7 - y1 
    fit3 = np.concatenate((y1,y4),axis=1) 
    fit3 = fit3[:2] 
    fits = np.concatenate((fit1,fit2,fit3), axis=0) 
    pops = np.arange(fits.shape[0]).reshape(fits.shape[0],1) 

    
    random.seed(123)

    indices = np.arange(fits.shape[0])
    random.shuffle(indices)
    fits = fits[indices]
    pops = pops[indices]
    print(indices) 


    ranks = nonDominationSort(pops, fits) 
    print('ranks:', ranks) 

    dis = crowdingDistanceSort(pops,fits,ranks) 
    print("dis:", dis) 


        









             
            
        



import random

from scipy import stats

def val560(li1, li2):
    """
    sample 560 examples from the list and calculate its pearson correlation
    input:
        li1: list of score
        li2: list of score
    output:
        pearsonr: pearson correlation
    """
    len_li1 = len(li1)
    len_li2 = len(li2)
    assert(len_li1 == len_li2)
    # sample the sid
    samples = random.sample(range(len_li1), 560)
    # extract the data from the lists
    sam_li1 = [li1[sample] for sample in samples]
    sam_li2 = [li2[sample] for sample in samples]
    # calcute the correaltion of the list
    pearsonr = stats.pearsonr(sam_li1, sam_li2)
    print (pearsonr)
    return pearsonr

def valTauLike(darr1, darr2):
    """
    calculate the Kendall's Tau like correlation
    input:
        f1: the list that save the human darr data, include 4 cols: sid, score1, score2, rr
        f2: the list that save the metric darr data
    output:
        taur: tau like correlation
    """
    assert(len(darr1) == len(darr2))
    # distribute the data into 9 blocks, 6 actuelly
    #print (darr1)
    #print (darr2)
    signs = [1, 0, -1]
    S = {}
    C = {}
    for i in signs:
        # i means the human scores
        if i == 0:
            continue
        for j in signs:
            # j is the metric scores
            C[str(i)+str(j)] = i*j
            S[str(i)+str(j)] = [1 for a, b in zip(darr1, darr2) if a == i and b == j]
    # calcute the tau like correlation
    numerator = 0
    denominator = 0
    for item in S.keys():
        #print (item)
        #print(C[item], len(S[item]))
        numerator += C[item] * len(S[item])
        denominator += len(S[item])
    if denominator == 0:
        print ('--- denominator is 0, set taul = 0')
        return 0
    print ("Numerator:",numerator)
    print ("Denominator:",denominator)
    taur = numerator*1.0/denominator
    return taur

def valPearsonFromFile(f1,f2):
    li1 = [float(li.rstrip('\n')) for li in open(f1)]
    li2 = [float(li.rstrip('\n')) for li in open(f2)]
    pearsonr = stats.pearsonr(li1, li2)
    print (pearsonr)


def valTauLikeFromFile(f1, f2):
    """
    input:
        f1: the file that save the human darr data, include 4 cols: sid, score1, score2, rr
        f2: the file that save the metric darr data
    """
    # get the darr data
    darr1 = [int(li.rstrip('\n').split(',')[-1]) for li in open(f1)]
    darr2 = [int(li.rstrip('\n').split(',')[-1]) for li in open(f2)]
    taur = valTauLike(darr1, darr2)
    return taur

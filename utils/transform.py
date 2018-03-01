import numpy as np

def daToRr(l1, l2, threshold = 0.1):
    """
    input:
        l1: the list of da scores for system output 1
        l2: the list of da scores for system output 2
        threshold: difference to distinguish two scores 
    output:
        out: list of rr scores, 1 means sys1 win, -1 means sys2 win, 0 means equal
    """
    assert(len(l1) == len(l2))
    lt = []
    for i in range(len(l2)):
        lt.append(threshold)
    out = map(_compare_num_with_threshold, l1, l2, lt)
    return out

def daToDarr(l1, l2):
    """
    input: 
        l1: the list of da scores for system output1
        l2: the list of da scores for system output2
    output:
        out: list of darr scores, 1 means sys1 win, -1 means sys2 win, 0 means equal
    """
    pass

def _convert_da_to_rank(da):
    return 5 - int(da/25)


def _compare_num_with_threshold(i1, i2, t):
    if i1 - i2 > t:
        return 1
    elif i1 - i2 < -t:
        return -1
    else:
        return 0



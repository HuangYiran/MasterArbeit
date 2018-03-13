import numpy as np

from collections import namedtuple

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

def daFileToRrFile(filename, dir_tgt = 'tmp/wmt17_ende.csv'):
    """
    input:
        filename: the name of da csv file
    """
    # read data and save it in the sents dict
    sents = {}
    Sysscore = namedtuple('Sysscore', ['sys', 'score'])
    with open(filename) as fi:
        for li in fi:
            items = li.rstrip('\n').split(' ')
            sys = items[0]
            sid = items[1]
            scr = items[2]
            ss = Sysscore(sys, scr)
            if sid in sents.keys():
                sents[sid].append(ss)
            else:
                sents[sid] = [ss]
    # transform the da data to rank data
    out = []
    for sid in sents.keys():
        out.extend([
            [
            'eng',
            'deu',
            sid,
            ss1.sys,
            ss2.sys,
            str(_getRank1(_compare_num_with_threshold(float(ss1.score), float(ss2.score), 25))),
            str(_getRank2(_compare_num_with_threshold(float(ss1.score), float(ss2.score), 25)))
            ]
            for i1, ss1 in enumerate(sents[sid])
            for i2, ss2 in enumerate(sents[sid])
            if i1 < i2
            ])
    # write data
    with open(dir_tgt, 'w') as fi:
        fi.write('srclang,trglang,segmentId,system1Id,system2Id,system1rank,system2rank\n')
        for li in out:
            record = ','.join(li) + '\n'
            fi.write(record)

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
    if i1 - i2 >= t:
        return 1
    elif i1 - i2 <= -t:
        return -1
    else:
        return 0

def _getRank1(scr):
    if scr == -1:
        return 2
    else:
        return 1

def _getRank2(scr):
    if scr == 1:
        return 2
    else:
        return 1


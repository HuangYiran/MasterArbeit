import argparse
import gensim
import string
import re
import os

import numpy as np
import nltk
#nltk.download('all')
from nltk.tokenize.moses import MosesTokenizer
"""
because the encoder embedding and decoder embedding data is to large. i can't save all the embeding data in the harddisk.
test3 is used to do the following tasks: read the word embedding vector and process same codes and for each sentence output only one vector.
"""
parser = argparse.ArgumentParser(description='test.py')


parser.add_argument('-ref', required=True,
                    help='Reference file')

parser.add_argument('-hyp', required=True,
                    help='Hypothesis')


parser.add_argument('-join', default="sum",
                    help='Word2VecModel')


def join(inp,add,op):
    if(op == "sum"):
        return inp + add
    elif(op == "max"):
        return np.maximum(inp,add)
    else:
        print ("Unknown operation:",op)
        exit()

def createVector(sentVec,op):
    if len(sentVec.shape) == 1:
        return sentVec
    num_words = sentVec.shape[0]
    sum = np.zeros(500)
    for i in range(num_words):
        sum = join(sum, sentVec[i], op)
    return sum
    
def distance(v1,v2):
    return np.absolute((v1 - v2)).sum()

def main():
    opt = parser.parse_args()

    # read the word embedding
    data_ref = np.load(opt.ref)
    data_hyp = np.load(opt.hyp)
    # assert the length
    assert(data_ref.shape[0]==data_hyp.shape[0])
    # calcute distance
    num_sent = data_ref.shape[0]
    for i in range(num_sent):
        rv = createVector(data_ref[i], opt.join)
        hv = createVector(data_hyp[i], opt.join)
        print ("Distance:", distance(rv,hv))
    
if __name__ == "__main__":
    main()

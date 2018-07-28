import argparse
import os
import sys
sys.path.append('./utils')
import nltk
import numpy as np

from valuation import valTauLike
#from normalize import normalize_minmax
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize.moses import MosesTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-s1', default = "./collect_data/data_s1", help = "the system output file")
parser.add_argument('-s2', default = "./collect_data/data_s2", help = "the second system output file")
parser.add_argument('-ref', default = "./collect_data/data_ref", help = "the reference file")
parser.add_argument('-src', default = "./collect_data/data_src", help = "the source file")
parser.add_argument('-scores', default = "./collect_data/data_scores", help = "the scores data")

def main():
    # load the parameter from the command line 
    opt = parser.parse_args()
    # run the chrF to get the chrF score for system one and system two
    print (">>> start getting the chrF scores")
    comm1 = 'cat ' + opt.s1 + ' |$MOSESROOT/scripts/tokenizer/tokenizer.perl > /tmp/tokenized_s1'
    comm2 = 'cat ' + opt.s2 + ' |$MOSESROOT/scripts/tokenizer/tokenizer.perl > /tmp/tokenized_s2'
    comm3 = 'cat ' + opt.ref + ' |$MOSESROOT/scripts/tokenizer/tokenizer.perl > /tmp/tokenized_ref'
    print(comm1)
    print(comm2)
    print(comm3)
    os.system(comm1)
    os.system(comm2)
    os.system(comm3)
    comm4 = 'cat /tmp/tokenized_s1 |$MOSESROOT/mert/sentence-bleu /tmp/tokenized_ref > ./tmp/tmp1'
    comm5 = 'cat /tmp/tokenized_s2 |$MOSESROOT/mert/sentence-bleu /tmp/tokenized_ref > ./tmp/tmp2'
    print(comm4)
    print(comm5)
    os.system(comm4)
    os.system(comm5)

    print ("<<< finish getting the chrF score and store then in tmp file")
    # extract the score from the file and remove the last three line of the tmp file, it contains the total valuation infos
    print (">>> read the score oben and compare the result")
    tmp_sc1 = [float(li.rstrip('\n').split('\t')[-1]) for li in open('./tmp/tmp1')]
    tmp_sc2 = [float(li.rstrip('\n').split('\t')[-1]) for li in open('./tmp/tmp2')]
    # conpare the score of two system
    assert(len(tmp_sc1) == len(tmp_sc2))
    def _compare(a,b):
        if a>b:
            return 1
        elif a<b:
            return -1
        else:
            return 0
    zip_sc = zip(tmp_sc1, tmp_sc2)
    tmp_rs = [_compare(sc1,sc2) for sc1, sc2 in zip_sc]
    print ("<<< finish comparing")
    # get the target data and calculate the tau like corr
    print (">>> read target data and calculate the tau like correlation")
    tgt_rs = [int(li.rstrip('\n')) for li in open(opt.scores)]
    taul = valTauLike(tgt_rs, tmp_rs)
    print ("<<< finish.")
    print (taul)


if __name__ == "__main__":
    main()


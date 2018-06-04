import argparse
import os
import sys
sys.path.append('./utils')
import re

from valuation import valTauLike
#from normalize import normalize_minmax
"""
remember to clean the data after runing the script, it takes lots of space
"""

parser = argparse.ArgumentParser()
parser.add_argument('-s1', default = "./collect_data/data_s1", help = "the system output file")
parser.add_argument('-s2', default = "./collect_data/data_s2", help = "the second system output file")
parser.add_argument('-ref', default = "./collect_data/data_ref", help = "the reference file")
parser.add_argument('-scores', default = "./collect_data/data_scores", help = "the scores data")
parser.add_argument('-src', default = "./collect_data/data_src", help = "the src file only used in decoder_hidden type")
parser.add_argument('-test_s1')
parser.add_argument('-test_s2')
parser.add_argument('-test_ref')
parser.add_argument('-test_scores')
parser.add_argument('-test_src')
parser.add_argument('-type', default = "decoder_embd", help = "type of word embeddings")
parser.add_argument('-mode', default = 'sum', help = "mode to precess the word embeddings")
parser.add_argument('-mode2', default = 'sum', help = "mode to precess the word embeddings")
parser.add_argument('-model', required = True, help = 'model to get the embeddings')

def main():
    # load the parameter from the command line 
    opt = parser.parse_args()
    # get lan pair and year and type
    get_date  = re.compile('2\d+')
    get_lan = re.compile('[a-z]{2}\.[a-z]{2}')
    date = get_date.search(opt.s1).group(0)
    lanp = get_lan.search(opt.s1).group(0)
    # set tmp name for the get_embeddings output
    o_ref = '/tmp/'+lanp+date+opt.type+'.ref'
    o_s1 = '/tmp/'+lanp+date+opt.type+'.s1'
    o_s2 = '/tmp/'+lanp+date+opt.type+'.s2'
    # run get_embeddings to get the word embeddings for the input
    if(not os.path.exists(o_ref) or not os.path.exists(o_s1) or not os.path.exists(o_s2)):
        comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.ref+" -output " + o_ref
        comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.s1+" -output " + o_s1
        comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.s2+" -output " + o_s2
        if opt.type == 'decoder_hidden' or opt.type == 'decoder_hidden_last':
            comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output " + o_ref +  " -tgt " + opt.ref
            comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output " + o_s1 + " -tgt " + opt.s1
            comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output " + o_s2 + " -tgt " + opt.s2

        print(comm_ref)
        print(comm_s1)
        print(comm_s2)
        os.system(comm_ref)
        os.system(comm_s1)
        os.system(comm_s2)

    # run test to get process the word embeddings
    print (">>> start getting the chrF scores")
    chrF1= "python code/NNMetric/test5.py -hyp " + o_s1 + " -ref " + o_ref + " > /tmp/tmp1"
    chrF2= "python code/NNMetric/test5.py -hyp " + o_s2 + " -ref " + o_ref + " >  /tmp/tmp2"
    print (chrF1)
    print (chrF2)
    os.system(chrF1)
    os.system(chrF2)
    print ("<<< finish getting the chrF score and store then in tmp file")

    print (">>> read the score oben and compare the result")
    tmp_sc1 = [float(li.rstrip('\n')) for li in open('/tmp/tmp1')]
    tmp_sc2 = [float(li.rstrip('\n')) for li in open('/tmp/tmp2')]
    #clean data
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

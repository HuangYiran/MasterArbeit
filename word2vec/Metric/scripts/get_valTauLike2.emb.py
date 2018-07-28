import argparse
import os
import sys
sys.path.append('./utils')

from valuation import valTauLike
#from normalize import normalize_minmax

parser = argparse.ArgumentParser()
parser.add_argument('-s1', default = "./collect_data/data_s1", help = "the system output file")
parser.add_argument('-s2', default = "./collect_data/data_s2", help = "the second system output file")
parser.add_argument('-ref', default = "./collect_data/data_ref", help = "the reference file")
parser.add_argument('-scores', default = "./collect_data/data_scores", help = "the scores data")
parser.add_argument('-src', default = "./collect_data/data_src", help = "the src file only used in decoder_hidden type")
parser.add_argument('-type', default = "decoder_embd", help = "type of word embeddings")
parser.add_argument('-mode', default = 'sum', help = "mode to precess the word embeddings")
parser.add_argument('-model', required = True, help = 'model to get the embeddings')

def main():
    # load the parameter from the command line 
    opt = parser.parse_args()
    # run get_embeddings to get the word embeddings for the input
    comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.ref+" -output /tmp/data_ref"
    comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.s1+" -output /tmp/data_s1"
    comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.s2+" -output /tmp/data_s2"
    if opt.type == 'decoder_hidden':
        comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output /tmp/data_ref -tgt " + opt.ref
        comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output /tmp/data_s1 -tgt " + opt.s1
        comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.src+" -output /tmp/data_s2 -tgt " + opt.s2

    print comm_ref
    print comm_s1
    print comm_s2
    os.system(comm_ref)
    os.system(comm_s1)
    os.system(comm_s2)

    # run test to get process the word embeddings
    print (">>> start getting the chrF scores")
    chrF1= "python code/NNMetric/test3.py -hyp /tmp/data_s1 -ref /tmp/data_ref -join " + opt.mode + " > ./tmp/tmp1"
    chrF2= "python code/NNMetric/test3.py -hyp /tmp/data_s2 -ref /tmp/data_ref -join " + opt.mode + " > ./tmp/tmp2"
#    chrF1= "python code/NNMetric/test.py -hyp " + opt.s1 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp1"
#    chrF2= "python code/NNMetric/test.py -hyp " + opt.s2 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp2"
    print (chrF1)
    print (chrF2)
    os.system(chrF1)
    os.system(chrF2)
    print ("<<< finish getting the chrF score and store then in tmp file")
    # extract the score from the file and remove the last three line of the tmp file, it contains the total valuation infos
    print (">>> read the score oben and compare the result")
    tmp_sc1 = [float(li.rstrip('\n')[1:-1].split(' ')[-1]) for li in open('./tmp/tmp1')]
    tmp_sc2 = [float(li.rstrip('\n')[1:-1].split(' ')[-1]) for li in open('./tmp/tmp2')]
    #clean data
    os.system('rm /tmp/data_ref')
    os.system('rm /tmp/data_s1')
    os.system('rm /tmp/data_s2')
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


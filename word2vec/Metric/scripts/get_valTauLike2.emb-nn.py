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
parser.add_argument('-test_s1', required = True)
parser.add_argument('-test_s2', required = True)
parser.add_argument('-test_ref', required = True)
parser.add_argument('-test_scores', required = True)
parser.add_argument('-test_src')
parser.add_argument('-type', default = "decoder_embd", help = "type of word embeddings")
parser.add_argument('-mode', default = 'sum', help = "mode to precess the word embeddings")
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
    
    # run get_embeddings to get the teset data
    test_date = get_date.search(opt.test_s1).group(0)
    test_lan = get_lan.search(opt.test_s1).group(0)
    # set tmp name for the get_embeddings output
    o_test_ref = '/tmp/'+test_lan+test_date+opt.type+'.ref'
    o_test_s1 = '/tmp/'+test_lan+test_date+opt.type+'.s1'
    o_test_s2 = '/tmp/'+test_lan+test_date+opt.type+'.s2'
    if(not os.path.exists(o_test_ref) or not os.path.exists(o_test_s1) or not os.path.exists(o_test_s2)):
        comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_ref+" -output " + o_test_ref
        comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_s1+" -output " + o_test_s1
        comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_s2+" -output " + o_test_s2
        if opt.type == 'decoder_hidden' or opt.type == 'decoder_hidden_last':
            comm_ref= "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_src+" -output " + o_test_ref + " -tgt " + opt.test_ref
            comm_s1 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_src+" -output " + o_test_s1 + " -tgt " + opt.test_s1
            comm_s2 = "python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/get_vector_from_sentence/get_embeddings.py -model "+opt.model+" -type "+opt.type+" -src "+opt.test_src+" -output " + o_test_s2 + " -tgt " + opt.test_s2

        print(comm_ref)
        print(comm_s1)
        print(comm_s2)
        os.system(comm_ref)
        os.system(comm_s1)
        os.system(comm_s2)

    # run test to get process the word embeddings
    print (">>> start getting the chrF scores")
    chrF1= "python code/NNMetric/test4.py -hyp " + o_s1 + " -join " + opt.mode + " -output /tmp/tmp1"
    chrF2= "python code/NNMetric/test4.py -hyp " + o_s2 + " -join " + opt.mode + " -output  /tmp/tmp2"
    chrF3= "Python code/NNMetric/test4.py -hyp " + o_ref + " -join " + opt.mode + " -output /tmp/tmp3"
#    chrF1= "python code/NNMetric/test.py -hyp " + opt.s1 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp1"
#    chrF2= "python code/NNMetric/test.py -hyp " + opt.s2 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp2"
    print (chrF1)
    print (chrF2)
    print (chrF3)
    os.system(chrF1)
    os.system(chrF2)
    os.system(chrF3)
    print ("<<< finish getting the chrF score and store then in tmp file")

    # run to get the test data
    chrF1= "python code/NNMetric/test4.py -hyp " + o_test_s1 + " -join " + opt.mode + " -output /tmp/test1"
    chrF2= "python code/NNMetric/test4.py -hyp " + o_test_s2 + " -join " + opt.mode + " -output /tmp/test2"
    chrF3= "Python code/NNMetric/test4.py -hyp " + o_test_ref + " -join " + opt.mode + " -output /tmp/test3"
#    chrF1= "python code/NNMetric/test.py -hyp " + opt.s1 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp1"
#    chrF2= "python code/NNMetric/test.py -hyp " + opt.s2 + " -ref " + opt.ref + "  -w2v data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz -join max > ./tmp/tmp2"
    print (chrF1)
    print (chrF2)
    print (chrF3)
    os.system(chrF1)
    os.system(chrF2)
    os.system(chrF3)
    print('train the model')
    #TODO assert
    o_png = '/tmp/'+lanp+date+opt.type+opt.mode+'_'
    commd = 'python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/experiments/run_expr_try.py  -doc /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/experiments/plan_c_rank_deen.xml -train_scores ' + opt.scores + ' -train_s1 /tmp/tmp1.npy -train_s2 /tmp/tmp2.npy -train_ref /tmp/tmp3.npy -test_scores ' + opt.test_scores + ' -test_s1 /tmp/test1.npy -test_s2 /tmp/test2.npy -test_ref /tmp/test3.npy -o_dir ' + o_png
    print(commd)
    os.system(commd)



if __name__ == "__main__":
    main()


import argparse
import os
import sys
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/utils')
import re

from valuation import valTauLike
"""
compare to one and two, this scripts is only writing for join method wr which are stored in test8
besides this script is only work when the training data are wmt1415 and testing data are wmt16
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
    chrF1 = "python code/NNMetric/test8.py -ref " + o_ref + " -s1 " + o_s1 + " -s2 " + o_s2 + " -test_ref " + o_test_ref + " -test_s1 " + o_test_s1 + " -test_s2 " + o_test_s2 + " -sent_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_1415/data_ref -sent_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_1415/data_s1 -sent_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_1415/data_s2 -sent_test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -sent_test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -sent_test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2"
    print (chrF1)
    os.system(chrF1)
    print('train the model')
    #TODO assert
    # extract the score from the file and remove the last three line of the tmp file, it contains the total valuation infos
    o_png = '/tmp/'+lanp+date+opt.type+opt.mode+'_'
    commd = 'python /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/experiments/run_expr_try.py  -doc /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/experiments/plan_c_rank_deen.xml -train_scores ' + opt.scores + ' -train_s1 /tmp/tmp1.npy -train_s2 /tmp/tmp2.npy -train_ref /tmp/tmp3.npy -test_scores ' + opt.test_scores + ' -test_s1 /tmp/test1.npy -test_s2 /tmp/test2.npy -test_ref /tmp/test3.npy -o_dir ' + o_png
    print(commd)
    os.system(commd)



if __name__ == "__main__":
    main()


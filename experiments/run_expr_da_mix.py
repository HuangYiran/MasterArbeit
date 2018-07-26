import sys
sys.path.append('utils/')
import nnXml as xml
import fmin2 as fmin
import Params
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-doc', default = 'fullHiddenModelTestList',
        help = "file to extract the parameter")
parser.add_argument('-getLast', action = 'store_true',
        help = "when set, only get the data of last hidden value")
parser.add_argument('-expr', default = 'plan_c_da_de.en',
        help = "the name of the experiment, it deside the data of the experiment")

opt = parser.parse_args()
# set the basic parameter: data source
if opt.expr == 'plan_a_en.de':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_a_en.de/train_scores',
          'src_sys': '../data/MasterArbeit/plan_a_en.de/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_a_en.de/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_a_en.de/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_a_en.de/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_a_en.de/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_a_en.de/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_a_en.de/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_a_en.de/test_ref_hidden',
            }
elif opt.expr == 'plan_a_de.en':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_a_de.en/train_scores',
          'src_sys': '../data/MasterArbeit/plan_a_de.en/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_a_de.en/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_a_de.en/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_a_de.en/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_a_de.en/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_a_de.en/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_a_de.en/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_a_de.en/test_ref_hidden',
            }
elif opt.expr == 'plan_b_en.de':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_b_en.de/train_scores',
          'src_sys': '../data/MasterArbeit/plan_b_en.de/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_b_en.de/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_b_en.de/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_b_en.de/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_b_en.de/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_b_en.de/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_b_en.de/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_b_en.de/test_ref_hidden',
            }
elif opt.expr == 'plan_b_de.en':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_b_de.en/train_scores',
          'src_sys': '../data/MasterArbeit/plan_b_de.en/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_b_de.en/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_b_de.en/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_b_de.en/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_b_de.en/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_b_de.en/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_b_de.en/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_b_de.en/test_ref_hidden',
            }
elif opt.expr == 'plan_c_rank_de.en':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_c_rank_de.en/train_scores',
          'src_sys': '../data/MasterArbeit/plan_c_rank_de.en/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_c_rank_de.en/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_c_rank_de.en/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_c_rank_de.en/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_c_rank_de.en/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_c_rank_de.en/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_c_rank_de.en/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_c_rank_de.en/test_ref_hidden',
            }
elif opt.expr == 'plan_c_rank_en.de':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_c_rank_en.de/train_scores',
          'src_sys': '../data/MasterArbeit/plan_c_rank_en.de/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/plan_c_rank_en.de/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/plan_c_rank_en.de/val_scores',
          'src_val_sys': '../data/MasterArbeit/plan_c_rank_en.de/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/plan_c_rank_en.de/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/plan_c_rank_en.de/test_scores',
          'src_test_sys': '../data/MasterArbeit/plan_c_rank_en.de/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/plan_c_rank_en.de/test_ref_hidden',
            }
elif opt.expr == 'plan_c_da_de.en':
    params_data = {
          'tgt': '../data/MasterArbeit/plan_c_da_de.en/training_scores',
          'src_sys': '../data/MasterArbeit/plan_c_da_de.en/hidden_training_sys',
          'src_ref': '../data/MasterArbeit/plan_c_da_de.en/hidden_training_ref',
          'tgt_val': '../data/MasterArbeit/plan_c_da_de.en/training_scores',
          'src_val_sys': '../data/MasterArbeit/plan_c_da_de.en/hidden_training_sys',
          'src_val_ref': '../data/MasterArbeit/plan_c_da_de.en/hidden_training_ref',
          'tgt_test': '../data/MasterArbeit/plan_c_da_de.en/testing_scores',
          'src_test_sys': '../data/MasterArbeit/plan_c_da_de.en/hidden_testing_sys',
          'src_test_ref': '../data/MasterArbeit/plan_c_da_de.en/hidden_testing_ref',
            }
elif opt.expr == 'word2vec':
    params_data = {
            'tgt': './word2vec/hidden_data/deen/train_scores',
            'src_sys': './word2vec/hidden_data/deen/train_sys',
            'src_ref': './word2vec/hidden_data/deen/train_ref',
            'tgt_val': './word2vec/hidden_data/deen/train_scores',
            'src_val_sys': './word2vec/hidden_data/deen/train_sys',
            'src_val_ref': './word2vec/hidden_data/deen/train_ref',
            'tgt_test': './word2vec/hidden_data/deen/test_scores',
            'src_test_sys': './word2vec/hidden_data/deen/test_sys',
            'src_test_ref': './word2vec/hidden_data/deen/test_ref'
            }

params_list = xml.read_exp_list(opt.doc)

results = []
for params in params_list:
    params.update(params_data)
    result = fmin.o_func(params)
    results.append(result)
print results

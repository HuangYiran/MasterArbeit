import sys
sys.path.append('utils/')
import nnXml as xml
import fmin
import Params
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-doc', default = 'fullHiddenModelTestList',
        help = "file to extract the parameter")
parser.add_argument('-getLast', action = 'store_true',
        help = "when set, only get the data of last hidden value")

opt = parser.parse_args()
# set same basic parameter
if opt.getLast:
    # last hidden value 
    params_data = {
          'tgt': '../data/MasterArbeit/test/train_scores',
          'src_sys': '../data/MasterArbeit/test/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/test/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/test/val_scores',
          'src_val_sys': '../data/MasterArbeit/test/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/test/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/test/test_scores',
          'src_test_sys': '../data/MasterArbeit/test/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/test/test_ref_hidden',
    }
else:
    # full hidden value
    params_data = {
          'tgt': '../data/MasterArbeit/test2/train_scores',
          'src_sys': '../data/MasterArbeit/test2/train_sys_hidden',
          'src_ref': '../data/MasterArbeit/test2/train_ref_hidden',
          'tgt_val': '../data/MasterArbeit/test2/val_scores',
          'src_val_sys': '../data/MasterArbeit/test2/val_sys_hidden',
          'src_val_ref': '../data/MasterArbeit/test2/val_ref_hidden',
          'tgt_test': '../data/MasterArbeit/test2/test_scores',
          'src_test_sys': '../data/MasterArbeit/test2/test_sys_hidden',
          'src_test_ref': '../data/MasterArbeit/test2/test_ref_hidden',
    }

params_list = xml.read_exp_list(opt.doc)

results = []
for params in params_list:
    params.update(params_data)
    result = fmin.o_func(params)
    results.append(result)
print results

import sys
sys.path.append('utils/')
import nnXml as xml
import fmin
import Params
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-doc', default = 'fullHiddenModelTestList',
        help = "file to extract the parameter")
parser.add_argument('-train_scores', required = True)
parser.add_argument('-train_s1', required = True)
parser.add_argument('-train_s2', required = True)
parser.add_argument('-train_ref', required = True)
parser.add_argument('-test_scores', required = True)
parser.add_argument('-test_s1', required = True)
parser.add_argument('-test_s2', required = True)
parser.add_argument('-test_ref', required = True)
parser.add_argument('-o_dir', default = '/tmp/1.png')

opt = parser.parse_args()
# set the basic parameter: data source
params_data = {
      'tgt': opt.train_scores,
      'src_sys': opt.train_s1,
      'src_sys2': opt.train_s2,
      'src_ref': opt.train_ref,
      'tgt_val': opt.train_scores,
      'src_val_sys': opt.train_s1,
      'src_val_sys2': opt.train_s2,
      'src_val_ref': opt.train_ref,
      'tgt_test': opt.test_scores,
      'src_test_sys': opt.test_s1,
      'src_test_sys2': opt.test_s2,
      'src_test_ref': opt.test_ref,
      'dir_mid_result': opt.o_dir,
        }

params_list = xml.read_exp_list(opt.doc)

results = []
for params in params_list:
    params.update(params_data)
    result = fmin.o_func(params)
    results.append(result)
print(results)

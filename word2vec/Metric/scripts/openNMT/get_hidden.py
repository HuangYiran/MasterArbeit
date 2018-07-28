from __future__ import division

import sys
sys.path.append("/project/wmt2012/user/jniehues/Metric/scripts/openNMT/OpenNMT-py/")
import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
from Pipeline_hidden import Pipeline_hidden

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

#parser.add_argument('-model', default = "../data/mt_model/prepro_model_ppl_20.07_e13.pt",
#                    help='Path to model .pt file')
parser.add_argument('-model', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen_ppl_e13.pt",
        help = 'Path to model .pt file')
parser.add_argument('-src', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/data/preprosrc-ref.bpe.noUndo.en",
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-tgt', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/data/preprosrc-ref.bpe.noUndo.de",
                    help='True target sequence (optional)')
parser.add_argument('-output', default="./test_data/hidden_value",
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")



parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-normalize', action='store_true',
                    help='To normalize the scores based on output length')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")

parser.add_argument('-get_last', action = 'store_true',
                    help = 'only get the hidden value of the last word in the sentence, when false, get the hidden value for each word in the sentence')

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    
    # Always pick n_best
    opt.n_best = opt.beam_size

    
    if opt.output == "stdout":
            outF = sys.stdout
    else:
            outF = open(opt.output, 'w')


    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    
    # here we are trying to open the file
    inFile = None
    if(opt.src == "stdin"):
            inFile = sys.stdin
            opt.batch_size = 1
    else:
      inFile = open(opt.src)

    pipeline = Pipeline_hidden(opt)

    for line in addone(inFile):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                # ???
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

    if opt.get_last:
        print("get last hidden value")
        decOut = pipeline.get_hidden(srcBatch, tgtBatch)
    else:
        print("get full hidden value")
        decOut = pipeline.get_hidden_full(srcBatch, tgtBatch)
    print(decOut.data.cpu().numpy().shape)
    
    with open(opt.output, "w") as f:
        tmp = decOut.data.cpu().numpy()
        numpy.save(f, tmp)

if __name__ == "__main__":
    main()

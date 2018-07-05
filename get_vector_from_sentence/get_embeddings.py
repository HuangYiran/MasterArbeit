from __future__ import division

import sys
sys.path.append("../OpenNMT-py")
sys.path.append("/Users/ihuangyiran/Documents/Workplace_Python/OpenNMT-py")
import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
from Pipeline_hidden import Pipeline_hidden

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required = True, default = '../data/mt_model/model_deen', help = 'Path to model.pt file')
parser.add_argument('-src', required = True, help = 'source sentence file')
parser.add_argument('-tgt', default = None, help = 'target sentence file if necessary')
parser.add_argument('-type', default = 'decoder_embd', help = 'set the target type: encoder_embd, decoder_embd, encoder_hidden, decoder_hidden')
parser.add_argument('-output', default = './', help = 'path to save the output')

parser.add_argument('-gpu', type = int, default = -1, help = "device to run on")
parser.add_argument('-batch_size', type = int, default = 30, help = 'batch size')
parser.add_argument('-max_sent_length', type = int, default = 100, help = 'maximum sentence length')
parser.add_argument('-replace_unk', action='store_true', help = """...""")
parser.add_argument('-verbose', action = 'store_true', help = 'Print scores and predictions for each sentence')
parser.add_argument('-num_dim', type = int, default = 500)

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt= parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    outF = open(opt.output, 'w')
    srcF = open(opt.src)
    tgtF = open(opt.tgt) if opt.tgt else None
    srcBatch, tgtBatch = [], []
    pipeline = Pipeline_hidden(opt)
    # transform
    for line in addone(srcF):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break
    # get the embedding
    if opt.type == 'encoder_embd':
        embs = pipeline.get_encoder_embedding(srcBatch)
    elif opt.type == 'decoder_embd':
        embs = pipeline.get_decoder_embedding(srcBatch)
    elif opt.type == 'encoder_hidden':
        embs = pipeline.get_encoder_output(srcBatch)
    elif opt.type == 'decoder_hidden':
        if opt.tgt:
            embs = pipeline.get_hidden_full(srcBatch, tgtBatch)
        else:
            print 'tgt file not set'
            embs = None
    elif opt.type == 'decoder_hidden_last':
        if opt.tgt:
            embs = pipeline.get_hidden(srcBatch, tgtBatch)
        else:
            print 'tgt file not set'
            embs = None
    elif opt.type == 'decoder_hidden_mean':
        if opt.tgt:
            embs = pipeline.get_hidden_mean(srcBatch, tgtBatch)
        else:
            print 'tgt file not set'
            embs = None
    elif opt.type == 'decoder_hidden_sum':
        if opt.tgt:
            embs = pipeline.get_hidden_sum(srcBatch, tgtBatch)
        else:
            print ' tgt file not set'
            embs = None
    elif opt.type == 'decoder_states':
        if opt.tgt:
            embs = pipeline.get_hidden_states(srcBatch, tgtBatch)
        else:
            print 'tgt file not set'
            embs = None
    elif opt.type == 'decoder_ceil':
        if opt.tgt:
            embs = pipeline.get_hidden_ceils(srcBatch, tgtBatch)
        else:
            print 'tgt file not set'
            embs = None
    else:
        print 'unrecognize type of embedding'
        embs = None
    # save the embeddings
    numpy.save(outF, embs.numpy())
    # close the file
    outF.close()
    srcF.close()
    if opt.tgt: 
        tgtF.close()


if __name__ == "__main__":
    main()
    

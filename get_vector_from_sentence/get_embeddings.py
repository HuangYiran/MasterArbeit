from __future__ import division

import sys
sys.path.append("../OpenNMT-py")
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
parser.add_argument('-type', default = 'decoder', help = 'get the encoder oder decoder embeddings')
parser.add_argument('-output', default = './', help = 'path to save the output')

parser.add_argument('-gpu', type = int, default = -1, help = "device to run on")
parser.add_argument('-batch_size', type = int, default = 30, help = 'batch size')
parser.add_argument('-max_sent_length', type = int, default = 100, help = 'maximum sentence length')
parser.add_argument('-replace_unk', action='store_true', help = """...""")


def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt. parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    outF = open(opt.output, 'w')
    srcF = open(opt.src)
    srcBatch, tgtBatch = [], []
    pipeline = Pipeline_hidden(opt)
    # transform
    for line in add(srcF):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break
    # get the embedding
    if opt.type == 'encoder':
        embs = pipeline.get_encoder_embedding(srcBatch)
    elif opt.type == 'decoder':
        embs = pipeline.get_decoder_embedding(srcBatch)
    else:
        print 'unrecognize type of embedding'
        embs = None
    # save the embeddings
    numpy.save(outF, embs.numpy())

if __name__ == "__main__":
    main()
    

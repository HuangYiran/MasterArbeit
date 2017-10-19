# -*- coding: UTF-8 -*-

import sys
sys.path.append("../OpenNMT-py/")
import onmt
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable

class Pipeline_hidden(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        if opt.verbose:
            print('Loading model from %s' % opt.model)
        checkpoint = torch.load(opt.model,
                               map_location=lambda storage, loc: storage)

        if opt.verbose:
            print('Done')

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        self._type = model_opt.encoder_type \
            if "encoder_type" in model_opt else "text"

        if self._type == "text":
            encoder = onmt.Models.Encoder(model_opt, self.src_dict)
        elif self._type == "img":
            loadImageLibs()
            encoder = onmt.modules.ImageEncoder(model_opt)

        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        model = onmt.Models.NMTModel(encoder, decoder)

        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        if self._type == "text":
            srcData = [self.src_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD)
                       for b in srcBatch]
        elif self._type == "img":
            srcData = [transforms.ToTensor()(
                Image.open(self.opt.src_img_dir + "/" + b[0]))
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type=self._type, balance=False)

    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)

    def get_hidden_batch(self, srcBatch, tgtBatch):
        # 1) run the encoder on the src
        # 其中encoder的输入的size是：[seq_len, batch, input_size], 这里seq_len即是numWords
        # 其中context是[seq_len, batch, hidden_size * num_directions]
        # encStates是一个tupel，他包括:
        #  - h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        #  - c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
        encStates, context = self.model.encoder(srcBatch)

        # 这里的srcBatch原本是dataset的输出，所以应该是(src, lengths)，下面这一步取出其中src的内容。
        srcBatch = srcBatch[0]
        batchSize = self._getBatchSize(srcBatch)

        # 获得RNN每层的节点数
        rnnSize = context.size(2)

        # 转换encState的维度为：[layers * batch * (directions * dim)]
        encStates = (self.model._fix_enc_hidden(encStates[0]), self.model._fix_enc_hidden(encStates[1]))

        decoder = self.model.decoder
        attentionLayer = decoder.attn

        # 如果类型为text，且batchSize大于1，则使用mask，原因未知？？？？？
        # 这个mask 将被用于decoder中的Attention，目的是使得attention能够忽略掉输入句子的padding的部分内容。
        # 但为什么仅在batchSize大于1的时候，才使用呢？？？
        useMasking = (self._type == 'text' and batchSize > 1)
        padMask = None
        if useMasking:
            padMask = srcBatch.data.eq(onmt.Constants.PAD).t() #标记pad的内容

        def mask(padMask):
            if useMasking:
                attentionLayer.applyMask(padMask)

        decStates = encStates
        # 初始化一个decoder的输出，
        decOut = self.model.make_init_decoder_output(context)
        mask(padMask)
        initOutput = self.model.make_init_decoder_output(context)
        # decoder的输出使outputs, hidden, atten, 前两者同一般的rnn输出，atten是nn.softmax()的输出
        # 关于tgtBatch，应该是一个size: [numWords, batchSize]的Variable(Datasetl类里面进行了转换)
        # globalAttention的参数是input: batch x hidden_size，context: batch x seq_len x hidden_size。
        # 对应输出的attn是batch X seq_len
        # 所以decoder的输出output, hidden, attn的size应该分别是：
        # [seq_len X batch X hidden_size], [num_layers X batch X hidden_size], [batch X seq_len]
        decOut, decStates, attn = self.model.decoder(tgtBatch[:-1], decStates, context, initOutput)

        return decOut, decStates, attn

    def get_hidden(self, srcBatch, goldBatch):
        # 把单词转化成对应的index，然后放进dataset中进行包装
        dataset = self.buildData(srcBatch, goldBatch)
        # 获得第一个num batch
        nu_batch = len(dataset)
        src, tgt, indices = dataset[0]
        tmp , _, _ = self.get_hidden_batch(src, tgt)
        out = tmp[-1]
        for i in range(1, nu_batch):
            print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            # 扔到translateBatch方法里面进行翻译, 这里src，tgt都是tensor类型维度为batchSize*numWord
            decOut, decStates, attn = self.get_hidden_batch(src, tgt)
            tmp = decOut[-1]
            out = torch.cat((out, tmp), 0)
             
        return out, decStates, attn
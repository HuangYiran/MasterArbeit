# -*- coding: UTF-8 -*-

import sys
sys.path.append("../OpenNMT-py/")
import onmt
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable
#from torch.nn.utils.rnn import pad_sequence

class ELMoMetric(torch.nn.Module):
    def __init__(self, opt):
        super(ELMoMetric, self).__init__()
        self.upStream = UpStream(opt)
        self.downStream = downStream()
    
    def forward(self, input):
        return self.downStream(self.upStream(input))

class UpStream(torch.nn.Module):
    def __init__(self, opt):
        super(Upstream, self).__init__()
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model,
                               map_location=lambda storage, loc: storage)
        # read model opt
        model_opt = checkpoint['opt']
        # load dict and type
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        self._type = model_opt.encoder_type \
            if "encoder_type" in model_opt else "text"
        # init encoder and decoder with dict
        if self._type == "text":
            self.encoder = onmt.Models.Encoder(model_opt, self.src_dict)
        elif self._type == "img":
            loadImageLibs()
            self.encoder = onmt.modules.ImageEncoder(model_opt)
        self.decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        # build NMTModel with encoder and decoder
        model = onmt.Models.NMTModel(self.encoder, self.decoder)
        # build generator ??? we don't need it here
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
            nn.LogSoftmax())
        # load the parameter for the model and generator
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        # if cuda
        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()
        model.generator = generator
        self.model = model
        # enter evaluation mode
        self.model.eval()

    def forward(self, srcBatch, tgtBatch):
        dataset = self.buildData(srcBatch, goldBatch)
        nu_batch = len(dataset)
        # get decoder hidden states
        out = []
        for i in range(nu_batch):
            #print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            # 扔到translateBatch方法里面进行翻译, 这里src，tgt都是tensor类型维度为batchSize*numWord
            tmp = self.get_decoder_hidden_states(src, tgt)
            # 把次序调整回改变之前，还没有测试，所以并不清楚tmp最后的类型，按道理应该是list类型，
            # 所以这个数据应该还要进行处理，这个留到测试的时候再做了、、、、、、、、、、、、、、？？
            tmp = list(zip(*sorted(zip(tmp, indices), key = lambda x: x[-1])))[:-1] 
            # tmp is a list, len(tmp)==1, len(tmp[0])==30, len(tmp[0][0])==500
            # so dimension of tmp is (1,30,500)
            # here i want to squezze the the fist dimension, but it is a list, therefore i process in a stupy way.
            # tmp[0] should have the same result
            #print len(tmp[0][0])
            #tmp = [tmp[0][i] for i, j in enumerate(tmp[0])]
            tmp = torch.stack(tmp[0])
            #indices = torch.LongTensor(indices)
            #tmp = tmp.index_select(0, indices) # can not be used here the indices here is the order of the original range
            #print(tmp.data.size())
            out.append(tmp)
        out = torch.cat(out, 0)
        # get decoder ceil states
        out2 = []
        for i in range(nu_batch):
            #print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            # 扔到translateBatch方法里面进行翻译, 这里src，tgt都是tensor类型维度为batchSize*numWord
            tmp = self.get_decoder_ceil_states(src, tgt)
            # 把次序调整回改变之前，还没有测试，所以并不清楚tmp最后的类型，按道理应该是list类型，
            # 所以这个数据应该还要进行处理，这个留到测试的时候再做了、、、、、、、、、、、、、、？？
            tmp = list(zip(*sorted(zip(tmp, indices), key = lambda x: x[-1])))[:-1] 
            # tmp is a list, len(tmp)==1, len(tmp[0])==30, len(tmp[0][0])==500
            # so dimension of tmp is (1,30,500)
            # here i want to squezze the the fist dimension, but it is a list, therefore i process in a stupy way.
            # tmp[0] should have the same result
            #print len(tmp[0][0])
            #tmp = [tmp[0][i] for i, j in enumerate(tmp[0])]
            tmp = torch.stack(tmp[0])
            #indices = torch.LongTensor(indices)
            #tmp = tmp.index_select(0, indices) # can not be used here the indices here is the order of the original range
            #print(tmp.data.size())
            out2.append(tmp)
        out2 = torch.cat(out2, 0)
        # get decoder embeddings
        out3 = get_decoder_embedding(tgt)
        #print(out.data.size())
        return out.data, out2.data, out3.data
    
    def get_decoder_embedding(self, tgtBatch):
        """
        input:
            tgtBatch: list of list of string
        output:
            out: type of tensor (batch_size, max_len, 500)
        """
        # convert to idx: tgtBatch is list of LongTensor
        tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in tgtBatch]
        # get the decoder embedding
        decoder_embeddings = self.decoder.word_lut.weight.data # type of Float Tensor (n_of_w, 500)
        tgtEmb = []
        for Idx in tgtData:
            tgtEmb.append(torch.index_select(decoder_embeddings, 0, Idx))
        # padding: out (batch_size, max_len, 500)
        out = self._pad_sequence(tgtEmb, batch_first = True)
        print out.shape
        # return the data
        return out
    
    def get_decoder_hidden_states(self, srcBatch, tgtBatch):
        # 原则上encoder的部分应该提到前面的，因为对每个句子这个操作是相同的，放在这里纯粹就是因为懒得改而已
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
        decOut = self.model.make_init_decoder_output(context) # ???
        mask(padMask)
        initOutput = self.model.make_init_decoder_output(context)
        # decoder的输出使outputs, hidden, atten, 前两者同一般的rnn输出，atten是nn.softmax()的输出
        # 关于tgtBatch，应该是一个size: [numWords, batchSize]的Variable(Datasetl类里面进行了转换)
        # globalAttention的参数是input: batch x hidden_size，context: batch x seq_len x hidden_size。
        # 对应输出的attn是batch X seq_len
        # 所以decoder的输出output, hidden, attn的size应该分别是：
        # [seq_len X batch X hidden_size], [num_layers X batch X hidden_size], [batch X seq_len]
        # 不懂这个-1在这里是什么意思，tgtBatch的维度是[batch_size, seq_len]，也就是说要少看一个句子的意思吗？这么做有什么意义，
        # 而且返回的时候的维度大小真的没有受影响吗？？？？？？？？？？、、、、、、、、？？？？？
        decOut, decStates, attn = self.model.decoder(tgtBatch[:-1], decStates, context, initOutput)
        return decStates[0] # tuple: batch_size, seq_len, num_layers, num_dim
    
    def get_decoder_hidden_states(self, srcBatch, tgtBatch):
        # 原则上encoder的部分应该提到前面的，因为对每个句子这个操作是相同的，放在这里纯粹就是因为懒得改而已
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
        decOut = self.model.make_init_decoder_output(context) # ???
        mask(padMask)
        initOutput = self.model.make_init_decoder_output(context)
        # decoder的输出使outputs, hidden, atten, 前两者同一般的rnn输出，atten是nn.softmax()的输出
        # 关于tgtBatch，应该是一个size: [numWords, batchSize]的Variable(Datasetl类里面进行了转换)
        # globalAttention的参数是input: batch x hidden_size，context: batch x seq_len x hidden_size。
        # 对应输出的attn是batch X seq_len
        # 所以decoder的输出output, hidden, attn的size应该分别是：
        # [seq_len X batch X hidden_size], [num_layers X batch X hidden_size], [batch X seq_len]
        # 不懂这个-1在这里是什么意思，tgtBatch的维度是[batch_size, seq_len]，也就是说要少看一个句子的意思吗？这么做有什么意义，
        # 而且返回的时候的维度大小真的没有受影响吗？？？？？？？？？？、、、、、、、、？？？？？
        decOut, decStates, attn = self.model.decoder(tgtBatch[:-1], decStates, context, initOutput)
        return decStates[1] # tuple: batch_size, seq_len, num_layers, num_dim
    
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
    def _pad_sequence(self, sequences, batch_first=False, padding_value=0):
        """
        ??? why we need to sort the sequence before we use this function????
        code from pytorch and change a bit.
        """
        max_len = max([x.size(0) for x in sequences])
        trailing_dims = sequences[0].size()[1:]
        #max_size = sequences[0].size()
        #max_len, trailing_dims = max_size[0], max_size[1:]
        prev_l = max_len
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # temporary sort check, can be removed when we handle sorting internally
            if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
            #prev_l = length
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
        return out_tensor

class DownStream(torch.nn.Module):
    def __init__(self, seq_len = 100, num_layers = 2, num_dim = 500):
        super(DownStream, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_dim = num_dim
        #self.weight = torch.autograd.Variable(torch.FloatTensor(self.num_layers, self.seq_len), requires_grad = True)
        self.weight_layers = torch.autograd.Variable(torch.FloatTensor(self.num_layers), requires_grad = True)
        self.weight_seq = torch.autograd.Variable(torch.FlaotTensor(self.seq_len), requires_grad = True)
        # build a mlp model
        self.mlp = nn.Sequential(
            torch.nn.Linear(1500, 500),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 3),
            torch.nn.LogSoftmax()
        )
        # softmax
        self.sf = torch.nn.Softmax
    
    def forward(self, input):
        # do the softmax for self.weight_layers, self.weight_seq
        weight_layers = self.sf(self.weight_layers)
        weight_seq = self.sf(sef.weight_seq)
        # expand weight so that it can do the bmm later, include self.weight_layers and self.weight_seq
        # get shape
        batch_size, seq_len, num_layers, num_dim = input[0].shape
        assert(seq_len == self.seq_len and num_layers = self.num_layers)
        exp_weight_layers = weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layers)
        exp_weight_seq = weight_seq.expand(batch_size, self.seq_len) # ==> (batch_size, seq_len)
        # mul input
        # use inner state only
        data = input[0]
        # weighted sum layer
        data = data.transpose(1,2)
        data = data.view(batch_size, num_layers, -1) # ==> (batch_size, num_layers, seq_len * num_dim)
        ls_data = torch.bmm(exp_weight_layers, data).squeeze() # ==> (batch_size,seq_len * num_dim)
        ls_data = ls_data.view(batch_size, seq_len, -1)
        # weighted sum seq
        ss_data = torch.bmm(exp_weight_seq, ls_data).squeeze() # ==> (batch_size, num_dim)
        out = self.mlp(ss_data)
        return out
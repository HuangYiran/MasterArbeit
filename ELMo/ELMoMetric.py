# -*- coding: UTF-8 -*-

import sys
sys.path.append("/Users/ihuangyiran/Documents/Workplace_Python/OpenNMT-py/")
import onmt
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable
#from torch.nn.utils.rnn import pad_sequence


#######################################
# ELMOMetric model and UpStream model #
#######################################
class ELMoMetric(torch.nn.Module):
    def __init__(self, opt):
        super(ELMoMetric, self).__init__()
        self.upStream = UpStream(opt)
        for params in self.upStream.parameters():
            params.requires_grad = False
        #self.downStream = DS_ELM()
        self.downStream = DS_DecMean()
    
    def forward(self, src, s1, s2, ref):
        # distribute the data
        out_s1 = self.upStream(src, s1)
        out_s2 = self.upStream(src, s2)
        out_ref = self.upStream(src, ref)
        # combine the data
        assert(len(out_s1) == len(out_s2) and
              len(out_s1) == len(out_ref))
        
        return self.downStream(out_s1, out_s2, out_ref)

class UpStream(torch.nn.Module):
    def __init__(self, opt):
        super(UpStream, self).__init__()
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
        self.seq_len = opt.seq_len
        self.tgt_lengths = torch.zeros(opt.batch_size).view(-1,1)

    def forward(self, srcBatch, goldBatch):
        for index, li in enumerate(goldBatch):
            self.tgt_lengths[index] = len(li)
        dataset = self.buildData(srcBatch, goldBatch)
        nu_batch = len(dataset) # should be one here
        # get decoder hidden states
        out = []
        out_decOut = []
        out_decHS = []
        out_decCS = []
        for i in range(nu_batch):
            #print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            #print '*'*10
            #print len(src[0]), len(tgt), len(indices)
            # 扔到translateBatch方法里面进行翻译, 这里src，tgt都是tensor类型维度为batchSize*numWord
            decOut, decHS, decCS = self.get_decoder_states(src, tgt)
            #print decOut.shape, decHS.shape, decCS.shape
            # 把次序调整回改变之前，还没有测试，所以并不清楚tmp最后的类型，按道理应该是list类型，
            # 所以这个数据应该还要进行处理，这个留到测试的时候再做了、、、、、、、、、、、、、、？？
            tmp = list(zip(*sorted(zip(decOut, decHS, decCS, indices), key = lambda x: x[-1])))[:-1] 
            # tmp is a list, len(tmp)==1, len(tmp[0])==30, len(tmp[0][0])==500
            # so dimension of tmp is (1,30,500)
            # here i want to squezze the the fist dimension, but it is a list, therefore i process in a stupy way.
            # tmp[0] should have the same result
            #print len(tmp[0][0])
            #tmp = [tmp[0][i] for i, j in enumerate(tmp[0])]
            decOut = torch.stack(tmp[0])
            decHS = torch.stack(tmp[1])
            decCS = torch.stack(tmp[2])
            #print decOut.shape, decHS.shape, decCS.shape
            #indices = torch.LongTensor(indices)
            #tmp = tmp.index_select(0, indices) # can not be used here the indices here is the order of the original range
            #print(tmp.data.size())
            out_decOut.append(decOut)
            out_decHS.append(decHS)
            out_decCS.append(decCS)
        out_decOut = torch.cat(out_decOut, 0)
        out_decHS = torch.cat(out_decHS, 0)
        out_decCS = torch.cat(out_decCS, 0)
        # get decoder embeddings
        out_decEmbd = self.get_decoder_embedding(tgt)
        # out_decOut (batch_size, seq_len, num_dim)
        # out_decHS (batch_size, seq_len, num_layer, num_dim)
        # out_decCS (batch_size, seq_len, num_layer, num_dim)
        # self.tgt_lengths (batch_size)
        return out_decOut, out_decHS, out_decCS, out_decEmbd, self.tgt_lengths
    
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
    
    def get_decoder_embedding(self, tgtBatch):
        """
        input:
            tgtBatch: torch.LongTensor of size (seq_len, batch_size)
        output:
            out: type of tensor (batch_size, max_len, 500)
        """
        # convert to idx: tgtBatch is list of LongTensor
        #tgtData = [self.tgt_dict.convertToIdx(b,
        #               onmt.Constants.UNK_WORD,
        #               onmt.Constants.BOS_WORD,
        #               onmt.Constants.EOS_WORD) for b in tgtBatch]
        # get the decoder embedding
        tgtEmb = []
        decoder_embeddings = self.decoder.word_lut.weight.data # type of Float Tensor (n_of_w, 500)
        for Idx in tgtBatch.data:
            tgtEmb.append(torch.index_select(decoder_embeddings, 0, Idx))
        # padding:
        tgtEmb = torch.stack(tgtEmb) # ==> (seq_len, batch_size, 500)
        #print('decEmbd shape befor padding: %s'%(str(tgtEmb.shape)))
        tgtEmb = tgtEmb.transpose(0,1)
        out = self._pad_sequence(tgtEmb, batch_first = True, padding_value = 0, max_len = self.seq_len)
        #print('decEmbd shape after padding: %s'%(str(tgtEmb.shape)))
        # return the data
        return out
    
    def get_decoder_states(self, srcBatch, tgtBatch):
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
        decOut = decOut.data.transpose(0,1) # ==> (batch_size, seq_len, num_dim)
        decHiddenStates = decStates[0].data.transpose(0,1)
        decCeilStates = decStates[1].data.transpose(0,1)
        decOut = self._pad_sequence(decOut, batch_first = True, padding_value = 0, max_len = self.seq_len)
        decHiddenStates = self._pad_sequence(decHiddenStates, batch_first = True, padding_value = 0, max_len = self.seq_len)
        decCeilStates = self._pad_sequence(decCeilStates, batch_first = True, padding_value = 0, max_len = self.seq_len)
        return decOut, decHiddenStates, decCeilStates # tuple: batch_size, seq_len, num_layers, num_dim
    
    def get_decoder_hidden_states(self, srcBatch, tgtBatch):
        # 还没有定句子长度，还是要改的
        _, decHiddenStates, _ = self.get_decoder_states(srcBatch, tgtBatch)
        return decHiddenStates
    
    def get_decoder_ceil_states(self, srcBatch, tgtBatch):
        _, _, decCeilStates = self.get_decoder_states(srcBatch, tgtBatch)
        return decCeilStates # tuple: batch_size, seq_len, num_layers, num_dim

    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)
    def _pad_sequence(self, sequences, batch_first=False, padding_value=0, max_len = None):
        """
        应该在transpose后进行pad操作，因为循环是用sequences长度的次数，batch_first值影响输出
        ??? why we need to sort the sequence before we use this function????
        code from pytorch and change a bit.
        """
        if max_len:
            max_len = max_len
        else:
            max_len = max([x.size(0) for x in sequences])
        trailing_dims = sequences[0].size()[1:]
        #max_size = sequences[0].size()
        #max_len, trailing_dims = max_size[0], max_size[1:]
        #prev_l = max_len
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # temporary sort check, can be removed when we handle sorting internally
            if max_len < length:
                #raise ValueError("lengths array has to be sorted in decreasing order")
                length = max_len
            #prev_l = length
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor[:length]
            else:
                out_tensor[:length, i, ...] = tensor[:length]
        return out_tensor

######################################
#         DownStream model           #
######################################
class DS_ELM(torch.nn.Module):
    def __init__(self, seq_len = 40, num_layers = 2, num_dim = 500):
        super(DS_ELM, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_dim = num_dim
        #self.weight = torch.autograd.Variable(torch.FloatTensor(self.num_layers, self.seq_len), requires_grad = True)
        self.weight_layers = torch.autograd.Variable(torch.FloatTensor(self.num_layers), requires_grad = True)
        self.weight_seq = torch.autograd.Variable(torch.FloatTensor(self.seq_len), requires_grad = True)
        # build a mlp model
        self.mlp = nn.Sequential(
            torch.nn.Linear(1500, 500),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 3),
            torch.nn.LogSoftmax()
        )
        # softmax
        self.sf = torch.nn.Softmax()
    
    def forward(self, out_s1, out_s2, out_ref):
        # only use decoder hidden states
        #print('out_s1[1] %s \nout_s2[1] %s \nout_ref[1] %s'%(str(out_s1[1].shape), str(out_s2[1].shape), str(out_ref[1].shape)))
        input = [out_s1[1], out_s2[1], out_ref[1]]
        input = torch.cat(input, 3) # (batch, seq_len, num_layer, num_dims)
        input = torch.autograd.Variable(input, requires_grad = False)
        # print input.data.shape
        # expand weight so that it can do the bmm later, include self.weight_layers and self.weight_seq
        # get shape
        batch_size, seq_len, num_layers, num_dim = input.data.shape
        assert(seq_len == self.seq_len and num_layers == self.num_layers)
        exp_weight_layers =self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layers)
        exp_weight_seq = self.weight_seq.expand(batch_size, self.seq_len) # ==> (batch_size, seq_len)
        # do the softmax for self.weight_layers, self.weight_seq
        exp_weight_layers = self.sf(exp_weight_layers).unsqueeze(1) # ==> (batch_size, 1, num_layers)
        exp_weight_seq = self.sf(exp_weight_seq).unsqueeze(1) # ==> (batch_size, 1, seq_len)
        # mul input
        # use inner state only
        data = input
        # weighted sum layer
        data = data.transpose(1,2).contiguous()
        data = data.view(batch_size, num_layers, -1) # ==> (batch_size, num_layers, seq_len * num_dim)
        ls_data = torch.bmm(exp_weight_layers, data).squeeze() # ==> (batch_size, seq_len * num_dim)
        ls_data = ls_data.view(batch_size, seq_len, -1) # ==> (batch_size, seq_len, num_dim)
        # weighted sum seq
        ss_data = torch.bmm(exp_weight_seq, ls_data).squeeze() # ==> (batch_size, num_dim)
        out = self.mlp(ss_data)
        return out

class DS_DecMean(torch.nn.Module):
    def __init__(self, seq_len = 40, num_layers = 2, num_dim = 500):
        super(DS_DecMean, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_dim = num_dim
        # build a mlp model
        self.mlp = nn.Sequential(
            torch.nn.Linear(1500, 500),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 3),
            torch.nn.LogSoftmax()
        )
        # softmax
        self.sf = torch.nn.Softmax()
        self.dsf = torch.autograd.Variable(torch.ones(123))
    
    def forward(self, out_s1, out_s2, out_ref):
        # only use decoder output value 
        #print('out_s1[1] %s \nout_s2[1] %s \nout_ref[1] %s'%(str(out_s1[1].shape), str(out_s2[1].shape), str(out_ref[1].shape)))
        inp_s1 = out_s1[0]
        inp_s2 = out_s2[0]
        inp_ref = out_ref[0]
        # get sent lengths
        len_s1 = out_s1[-1]
        len_s2 = out_s2[-1]
        len_ref = out_ref[-1]
        # get the num_dim
        batch_size, seq_len, num_dim = inp_s1.shape
        # expand the len data, so it can be divide by torch
        assert(len(len_s1) == len(len_s2), len(len_s1) == len(len_ref))
        if len(len_s1) > batch_size:
            len_s1 = len_s1[:batch_size]
            len_s2 = len_s2[:batch_size]
            len_ref = len_ref[:batch_size]
        len_s1 = len_s1.expand(batch_size, num_dim).type(torch.FloatTensor)
        len_s2 = len_s2.expand(batch_size, num_dim).type(torch.FloatTensor)
        len_ref = len_ref.expand(batch_size, num_dim).type(torch.FloatTensor)
        # calculate the mean of the input through the seq_len dimmension
        inp_s1 = inp_s1.sum(1)/len_s1
        inp_s2 = inp_s2.sum(1)/len_s2
        inp_ref = inp_ref.sum(1)/len_ref
        # combine the input data
        input = [inp_s1, inp_s2, inp_ref] # (batch_size, num_dim) * 3
        input = torch.cat(input, 1) # (batch, 3 * num_dims)
        input = torch.autograd.Variable(input, requires_grad = False)
        # run the mlp
        out = self.mlp(input)
        return out
        
    
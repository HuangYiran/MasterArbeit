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
    
    def _get_sent_length(self, src):
        """
        获得batch中每个句子的真实长度
        input: 
            src: [numWord, batch_size]
        out: 
            lengths: list [batch_size]
        """
        lengths = []
        length = 0
        data = src.t()
        #print(src)
        for line in data:
            #print(line)
            for label, word in enumerate(line):
                length = label
                #print(type(word.data[0]), str(word.data[0]))
                #print(type(onmt.Constants.PAD), onmt.Constants.PAD)
                if word.data[0] == onmt.Constants.PAD:
                    break;
            # 第length个值是pad_word，所以最后一个再其前一个
            # 另外这里的长度是指下标，所以下面还是进行了减一。
            # 然而存在另外一个问题就是，如果长度为0的话，那么下标会是-1，
            if length != 0:
                lengths.append(length - 1)
            else:
                lengths.append(length)
        return lengths
                    

    def get_hidden_batch(self, srcBatch, tgtBatch):
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
        decOut = self.model.make_init_decoder_output(context)
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
        decOut = decOut.transpose(0, 1).contiguous()

        return decOut

    def _get_last_hidden(self, src, tgt):
        pred = self.get_hidden_batch(src, tgt) # 输出pred维度为[batch_size, seq_len, hidden_size]
        #print(pred)
        # 对于每个句子，我们并不需要seq_len个数据，我们只需要最后一个单词对应的中间层输出，但是注意这里的句子都是经过padding处理的。
        # 所以对每个句子，我们应该根据goldBatch而不是tgt的长度进行，但是由于分batch和排序的原因，使用goldBatch进行处理其实是不太现实的，
        # 所以这里的方法就是，识别填充符，手动确认句子的长度
        lengths_tgt = self._get_sent_length(tgt)
        #print("lengths:----------")
        #print(lengths_tgt)
        tmp = []
        for counter, item in enumerate(pred):
            # item 是[seq_len, hidden_size]，得到的tmp是list其中每一项为[1, hidden_size]
            tmp.append(item[lengths_tgt[counter]])
        # 把list转化为[batch, hidden_size]
        tmp = torch.stack(tmp,0)
        #print(tmp)
        return tmp
    
    def _get_full_hidden(self, src, tgt):
        fixed_seq_len = 100
        pred = self.get_hidden_batch(src, tgt) # 输出pred维度为[batch_size, seq_len, hidden_size]
        # 每个Batch的seq_len是一样的，为这个Batch中最长的句子的长度。另外<EOS>后的hidden value虽然是不一样的，但是并没有价值
        # 下面的操作是为了提取有用的hidden value，并把结果扩充到固定的句子长度，这里设定为100
        lengths_tgt = self._get_sent_length(tgt)
        tmp = []
        for counter, item in enumerate(pred):
            # item 是 [seq_len, hidden_size], 得到的tmp是扩充后的item，应该是[100, hidden_size]
            num_word = lengths_tgt[counter] + 1 # get_sent_length返回的是坐标，所以加一
            if num_word >= fixed_seq_len:
                item_pad = item[:fixed_seq_len]
            else:
                item_ext = item[:num_word] # 加一是因为片取是不包括最后一位的。
                padding = torch.Tensor(fixed_seq_len - num_word, item.size()[1]).fill_(0)
                item_pad = torch.cat((item_ext, padding), 0)
            tmp.append(item_pad)
        tmp = torch.stack(tmp,0)
        return tmp
    
    def get_hidden(self, srcBatch, goldBatch):
        """
        获得每个句子隐藏层最后一个单词对应的输出
        """
        # 把单词转化成对应的index，然后放进dataset中进行包装
        dataset = self.buildData(srcBatch, goldBatch) #设定balance为False了，因为没有给输入进行排序
        # 获得第一个num batch
        nu_batch = len(dataset)
        out = []
        #for i in range(3):
        for i in range(nu_batch):
            print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            # 扔到translateBatch方法里面进行翻译, 这里src，tgt都是tensor类型维度为batchSize*numWord
            tmp = self._get_last_hidden(src, tgt)
            # 把次序调整回改变之前，还没有测试，所以并不清楚tmp最后的类型，按道理应该是list类型，
            # 所以这个数据应该还要进行处理，这个留到测试的时候再做了、、、、、、、、、、、、、、？？
            #tmp = list(zip(*sorted(zip(tmp, indices), key = lambda x: x[-1])))[:-1] 
            #tmp = [tmp[0][i] for i, j in enumerate(tmp[0])]
            #tmp = torch.stack(tmp)
            indices = torch.LongTensor(indices)
            tmp = tmp.index_select(0, indices)
            #print(tmp.data.size())
            out.append(tmp)
        out = torch.cat(out, 0)
        print(out.data.size())
        return out
    
    def get_hidden_full(self, srcBatch, goldBatch):
        """
        获得隐藏层的输出，
        output:
        out [batch_size, sql_len, hidden_size]
        """
        dataset = self.buildData(srcBatch, goldBatch)
        nu_batch = len(dataset)
        out = []
        for i in range(nu_batch):
            print("processing batch %s/%s" %(i, nu_batch))
            src, tgt, indices = dataset[i]
            tmp = self._get_full_hidden(src, tgt)
            #tmp = list(zip(*sorted(zip(tmp, indices), key = lambda x:x[-1])))[:-1]
            #tmp = [tmp[0][i] for i, j in enumerate(tmp[0])]
            #tmp = torch.stack(tmp)
            indices = torch.LongTensor(indices)
            tmp = tmp.index_select(0, indices)
            print(tmp.data.size())
            out.append(tmp)
        out = torch.cat(out, 0)
        print(out.data.size())
        return out
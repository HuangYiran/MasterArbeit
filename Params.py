class Params(object):
    def __init__(self):
        # public params
        self.model = './model/LinearModel'
        self.src_sys = '../data/MasterArbeit/data/hidden_value_pred'
        self.src_ref = '../data/MasterArbeit/data/hidden_value_ref'
        self.tgt = '../data/MasterArbeit/data/record_prepronce_clearned'
        self.src_test_sys = '../data/MasterArbeit/data/hidden_value_pred_2015'
        self.src_test_ref = '../data/MasterArbeit/data/hidden_value_ref_2015'
        self.tgt_test = '../data/MasterArbeit/data/record_newstest2015_cleaned'
        self.src_val_sys = '../data/MasterArbeit/data/hidden_value_pred_2016'
        self.src_val_ref = '../data/MasterArbeit/data/hidden_value_ref_2016'
        self.tgt_val = '../data/MasterArbeit/data/record_newstest2016_cleaned'
        self.resume = False
        self.checkpoint = './checkpoints/cp1'
        self.out = './pred'
        self.dir_mid_result = '../data/MasterArbeit/mid_result/'
        self.optim = 'Adam'
        self.loss_fn = 'MSELoss'
        self.batch_size = 50
        self.cuda = False
        self.isRandom = False
        self.debug = False
        
        # rank model, add an additional system output data
        self.rank = True
        self.src_sys2 = ''
        self.src_test_sys2 = ''
        self.src_val_sys2 = ''
        
        # PReLU loss function
        self.PReLU_rate = 2

        # linear model
        self.lr = 0.02
        self.eps = 1e-08
        self.weight_decay = 0
        self.dim2 = 500
        self.dim3 = None
        self.act_func = 'ReLU'
        self.act_func_out = None
        self.drop_out_rate = 0.5
        self.momentum = 0.1
        
        # FullHiddenAttn - mlp, rnn
        self.num_head = 8
        self.num_dim_k = 64
        self.num_dim_v = 64
        self.d_rate_attn = 0.1
        #self.dim2 = 20
        self.act_func1 = "LeakyReLU"
        self.act_func2 = "LeakyReLU"
        
        # FullHiddenAttn - conv
        self.dim1 = 20
        #self.act_func1 = "LeakyReLU"
        self.kernel_size1 = 3
        self.stride1 = 2
        self.kernel_size2 = 3
        self.stride2 = 2
    
    def set_params(self, opt):
        for name, value in opt.items():
            setattr(self, name, value)

    def show_params(self):
        for name, value in vars(self).items():
            print(name, value)
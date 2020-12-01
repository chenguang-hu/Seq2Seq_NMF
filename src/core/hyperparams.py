class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'data/src-train.txt'
    target_train = 'data/tgt-train.txt'
    source_test = 'data/src-test.txt'
    target_test = 'data/tgt-test.txt'
    source_dev = 'data/src-valid.txt'
    target_dev = 'data/tgt-valid.txt' 
    # training
    batch_size = 128 # alias = N
    lr = 1e-4 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'log_dir' # log directory
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <unk>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 500
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    patience = 5
    num_layers=1
    max_turn=15
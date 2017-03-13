import argparse
 
 
def str2bool(v):
    return v.lower() in ('y', 'yes', 't', 'true', '1')
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
 
    parser.add_argument('--random_seed',
                        type=int,
                        default=1013,
                        help='Random seed')
 
    parser.add_argument('--vocab_size',
                        type=int,
                        default=10000,
                        help='Default embed size')
 
    parser.add_argument('--embed_size',
                        type=int,
                        default=128,
                        help='Default embedding size if embedding_file is not given')
 
    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')
 
    parser.add_argument('--num_labels',
                        type=int,
                        default=96,
                        help='num labels')
 
    parser.add_argument('--bidir',
                        type='bool',
                        default=True,
                        help='bidir: whether to use a bidirectional RNN')
 
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of RNN layers')
 
    parser.add_argument('--rnn_type',
                        type=str,
                        default='gru',
                        help='RNN type: lstm or gru (default)')
 
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
 
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.2,
                        help='Dropout rate')
 
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer: sgd (default) or adam or rmsprop')
 
    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for SGD')
 
    return parser.parse_args()


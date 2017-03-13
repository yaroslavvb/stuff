import logging
import time
import config
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
 
 
def retrieve_seq_length_op2(data):
    return tf.reduce_sum(tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), 1)
 
 
def advanced_indexing_op(input, index):
    batch_size = tf.shape(input)[0]
    max_length = tf.shape(input)[1]
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant
 
 
def bidirectional_dynamic_rnn(inputs, cell_fn, n_hidden, sequence_length=None, return_last=False, name='bidyrnn'):
    with tf.variable_scope(name):
        batch_size = array_ops.shape(inputs)[0]
 
        fw_cell = cell_fn(num_units=n_hidden)
        bw_cell = cell_fn(num_units=n_hidden)
 
        fw_initial_state = fw_cell.zero_state(batch_size, dtype=tf.float32)
        bw_initial_state = bw_cell.zero_state(batch_size, dtype=tf.float32)
 
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=fw_initial_state,
            initial_state_bw=bw_initial_state,
        )
        outputs = tf.concat(outputs, 2)
        if return_last:
            outputs = advanced_indexing_op(outputs, sequence_length)
        return outputs
 
 
def BilinearAttention(inputs, n_hidden, mask=None,
                      initializer=tf.random_uniform_initializer(-0.01, 0.01)):
    W = tf.get_variable('W', shape=(n_hidden, n_hidden),
                        initializer=initializer)
    M = tf.matmul(inputs[1], W)
    M = tf.expand_dims(M, axis=1)
    alpha = tf.nn.softmax(tf.reduce_sum(inputs[0] * M, axis=2))
    if mask is not None:
        alpha *= mask
        alpha /= tf.reduce_sum(alpha, axis=1, keep_dims=True)
    alpha = tf.expand_dims(alpha, axis=2)
    outputs = tf.reduce_sum(inputs[0] * alpha, axis=1)
 
    return outputs
 
 
def inference(x1, x2, mask1, mask2, l, y,
              args, embeddings, reuse=False, training=False):
    with tf.variable_scope('model', reuse=reuse):
        embed = tf.get_variable('embed', shape=embeddings.shape,
                                initializer=tf.constant_initializer(embeddings))
        embed1 = tf.nn.embedding_lookup(embed, x1)
        embed2 = tf.nn.embedding_lookup(embed, x2)
 
        keep = 1.0 - args.dropout_rate if training else 1.0
        dropout1 = tf.nn.dropout(embed1, keep)
        dropout2 = tf.nn.dropout(embed2, keep)
 
        rnn_cell = {'gru': tf.contrib.rnn.GRUCell,
                    'lstm': tf.contrib.rnn.LSTMCell}[args.rnn_type]
        rnn1 = bidirectional_dynamic_rnn(dropout1, cell_fn=rnn_cell,
                                         n_hidden=args.hidden_size,
                                         sequence_length=retrieve_seq_length_op2(mask1),
                                         name='rnn1')
        rnn2 = bidirectional_dynamic_rnn(dropout2, cell_fn=rnn_cell,
                                         n_hidden=args.hidden_size,
                                         sequence_length=retrieve_seq_length_op2(mask2),
                                         return_last=True,
                                         name='rnn2')
 
        args.rnn_output_size = 2 * args.hidden_size
        att = BilinearAttention([rnn1, rnn2], args.rnn_output_size, mask1)
 
        z = tf.layers.dense(att, units=args.num_labels,
                            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                            use_bias=False)
 
        prob = tf.nn.softmax(z)
        prob = prob * l
        prob /= tf.reduce_sum(prob, axis=1, keep_dims=True)
 
        pred = tf.to_int32(tf.arg_max(prob, dimension=1))
        acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
 
        if not training:
            return acc
        else:
            epsilon = 1e-7
            prob = tf.clip_by_value(prob, epsilon, 1 - epsilon)
            loss = tf.one_hot(y, depth=args.num_labels) * -tf.log(prob)
            loss = tf.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(loss)
 
            if args.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer()
            elif args.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)
            else:
                raise NotImplementedError('optimizer = %s' % args.optimizer)
            train_op = optimizer.minimize(loss)
            return train_op, loss, acc
 
 
def main(args):
    logging.info('-' * 50)
    logging.info('Preparing data..')
 
    embeddings = np.random.uniform(-1, 1, (args.vocab_size, args.embed_size))
    x1 = np.random.choice(args.vocab_size, (4 * args.batch_size, 2000))
    x2 = np.random.choice(args.vocab_size, (4 * args.batch_size, 50))
    len1 = np.random.choice(np.arange(1000, 2000), 4 * args.batch_size)
    mask1 = np.ones((4 * args.batch_size, 2000)).astype('float32')
    for l, mask in zip(len1, mask1):
        mask[l:] = 0
    len2 = np.random.choice(np.arange(25, 50), 4 * args.batch_size)
    mask2 = np.ones((4 * args.batch_size, 50)).astype('float32')
    for l, mask in zip(len2, mask2):
        mask[l:] = 0
    l = np.ones((4 * args.batch_size, args.num_labels)).astype('float32')
    y = np.random.choice(args.num_labels, 4 * args.batch_size)
 
    in_x1 = tf.placeholder(tf.int32, [None, None])
    in_x2 = tf.placeholder(tf.int32, [None, None])
    in_mask1 = tf.placeholder(tf.float32, [None, None])
    in_mask2 = tf.placeholder(tf.float32, [None, None])
    in_l = tf.placeholder(tf.float32, [None, None])
    in_y = tf.placeholder(tf.int32, [None])
    feed_dict = {in_x1: x1, in_x2: x2,
                 in_mask1: mask1, in_mask2: mask2,
                 in_l: l, in_y: y}
 
    q = tf.RandomShuffleQueue(capacity=2000 * args.batch_size, min_after_dequeue=0,
                              dtypes=[tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32],
                              shapes=[x1.shape[1:], x2.shape[1:],
                                      mask1.shape[1:], mask2.shape[1:],
                                      l.shape[1:], y.shape[1:]])
    q_size = q.size()
    enqueue_op = q.enqueue_many([in_x1, in_x2, in_mask1, in_mask2, in_l, in_y])
    qr = tf.train.QueueRunner(q, [enqueue_op])
    all_data = q.dequeue_many(args.batch_size)
 
    logging.info('Building Computation Graph..')
    train_op, loss, ac = inference(*all_data, args, embeddings, reuse=False, training=True)
    light_op = tf.square(all_data[0])
 
    logging.info('-' * 50)
    logging.info('Create TensorFlow session..')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
 
    logging.info('Initialize model parameters..')
    sess.run(tf.global_variables_initializer())
 
    logging.info('-' * 50)
    logging.info(args)
 
    logging.info('-' * 50)
    logging.info('Start training..')
 
    num_samples_in_queue = sess.run(q_size)
    while num_samples_in_queue < 1999 * args.batch_size:
        sess.run(qr.enqueue_ops, feed_dict)
        num_samples_in_queue = sess.run(q_size)
        print("Recharging queue, current size = %i" % num_samples_in_queue)
 
    writer = tf.summary.FileWriter('summary', sess.graph)
    idx = 0
    while num_samples_in_queue > args.batch_size:
        idx += 1
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        begin_time = time.time()
        #sess.run(light_op, options=run_options, run_metadata=run_metadata)
        sess.run(train_op, options=run_options, run_metadata=run_metadata)
        end_time = time.time()
        writer.add_run_metadata(run_metadata, 'step = %d' % idx)
        writer.flush()
        num_samples_in_queue = sess.run(q_size)
        logging.info('%d left in queue' % num_samples_in_queue)
        logging.info('elapsed time %.2f(s)' % (end_time - begin_time))
 
    writer.close()
    logging.info('-' * 50)
    logging.info('Close TensorFlow session..')
    sess.close()
 
 
if __name__ == "__main__":
    args = config.get_args()
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
 
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
 
    main(args)

    

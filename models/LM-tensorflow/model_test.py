import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        # 如果处于测试阶段，那么一批有一个，一个长为1
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        ## 以下：搭建，以不同cell为底的，包裹有dropout的多层RNN网络结构--------------------------------------------------------------------------
        # choose different rnn cell
        # 选择基础的cell类型，根据args中的model参数
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        # num_layers 层 RNN
        for _ in range(args.num_layers):
            # rnn_size 一层内的隐藏单元数。
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                # dropout 就是一个somehow可以降低overfit程度的东西。把他包裹在cell外面即可。
                # 参数有两个，如下。给定自args
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            # 将其全部放入cells
            cells.append(cell)
        # 单个的cell组成cells，cells是list形态，将其叠堆在一起，变成MultiRNNCell。
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        ## -----------------------------------------------------------------------------------------------------


        ## 以下：声明input_data, targets, initial_state, softmax_w, softmax_b------------------------------------
        # input/target data (int32 since input is char-level)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            # rnn_size 代表一层内的rnn 单元数。
            # vocab_size 代表词语的数目。
            # softmax层是输出层。rnn cell内的w b 似乎都已经被集成好了。
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
        ## ----------------------------------------------------------------------------------------------------

        # transform input to embedding
        # 此处embedding也被训练。
        # 在 embedding 中 找 input_data。两个都是list，那么则依次寻找input_data的每一行。
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])

        # shape of inputs
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # 问题：这是啥？？inputs 的维度是多少？

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            # 维数大小不变，要么变成 0，要么变成1/output_keep_prob 倍，保证和不变。
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # unstack the input to fits in rnn model
        # axis=1 的意思是是dimension=1
        # 将其分为 seq_length 个小tensor
        # 问题：关于维数？以及matmul(prev, softmax_w)，维数？
        inputs = tf.split(inputs, args.seq_length, 1)
        # squeeze挤，把所有维度是1的向量消去。[1]代表只在原来维度=1的地方消去。
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            # 在输出层调用，softmax出来的概率之后，生成下一个的新概率。
            # prev_symbol得到概率最大的编号。
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        # 问题：在输出后连接1？
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # output layer
        # logits 意为 raw predictions。传入softmax层，做normalization
        # logits用于计算cross entropy 因为 ce 其中自带normalization
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # loss is calculate by the log loss and taking the average.
        # return the log-perplexity
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])], #此处的-1是自己计算之意，即让其变为一维向量。
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        # calculate gradients
        # 对于cost函数，对每个变量求导。加上clipper，得到一个按照 tvars 顺序排列的， 有每个变量gradient的序列：grads
        # nb之处在于tf是计算图，留存了每个节点的计算来源，可以直接通过变量名求导。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        # apply gradient change to the all the trainable variable.
        # 实际上给出 minimize(cost) 的含义也不过是对每个变量求导，最终back_prop修改其值。
        # 所以如果事先提供了每个变量的gradient，那么自然在这里只需要给出变量导数、变量名，就可以完成bp的工作了。
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

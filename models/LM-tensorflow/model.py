import numpy as np
import tensorflow as tf
import time

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from utils import SummaryHelper

class LMModel(object):
	def __init__(self, data, args, embed):
		self.args = args

		with tf.variable_scope("input"):
			with tf.variable_scope("embedding"):
			# build the embedding table and embedding input
				if embed is None:
					# initialize the embedding randomly
					self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
				else:
					# initialize the embedding by pre-trained word vectors
					self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

			self.raw_input_data = tf.placeholder( tf.int32, (None, None) ) # batch * len
			self.raw_input_data_length = tf.placeholder( tf.int32, (None,) ) # batch

			self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

			batch_size, batch_len = tf.shape(self.raw_input_data)[0], tf.shape(self.raw_input_data)[1]
			self.decoder_max_len = batch_len - 1

			self.input_data = tf.nn.embedding_lookup(self.embed, self.raw_input_data)  # batch*len*unit
			self.input_data_len = self.raw_input_data_length

			decoder_input = tf.split(self.raw_input_data, [self.decoder_max_len, 1], 1)[0]  # no eos_id
			self.decoder_input = tf.nn.embedding_lookup(self.embed, decoder_input)  # batch*(len-1)*unit
			self.decoder_target = tf.split(self.raw_input_data, [1, self.decoder_max_len], 1)[1]  # no go_id, batch*(len-1)
			self.decoder_len = self.raw_input_data_length - 1
			self.decoder_mask = tf.sequence_mask(self.decoder_len, self.decoder_max_len, dtype=tf.float32)  # batch*(len-1)

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build rnn_cell
		cellLSTM = tf.nn.rnn_cell.LSTMCell(args.dh_size)
		cellLSTMs = []
		cellLSTMs.append( cellLSTM )
		cell = rnn.MultiRNNCell(cellLSTMs, state_is_tuple=True)

		initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.variable_scope('rnnlm'):
			softmax_w = tf.get_variable("softmax_w", [args.dh_size, data.vocab_size])
			softmax_b = tf.get_variable("softmax_b", [data.vocab_size])

		self.input_max_len = batch_len - 1
		inputs = tf.split(self.input_data, [self.input_max_len, 1], 1)[0]

		def loop(prev, _):
			prev = tf.matmul(prev, softmax_w) + softmax_b
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, prev_symbol)
		# rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
		outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
		output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

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
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr)

		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
		tf.summary.histogram('logits', self.logits)
		tf.summary.histogram('loss', loss)
		tf.summary.scalar('train_loss', self.cost)

		# save checkpoint
		self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=args.checkpoint_max_to_keep, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
		self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

		# create summary for tensorboard
		self.create_summary(args)

	def store_checkpoint(self, sess, path, key):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step = self.global_step)
		else:
			self.best_saver.save(sess, path, global_step = self.global_step)
			#self.best_global_step = self.global_step

	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss",
																"perplexity",
																],
														prefix="train")

		scalarlist = ["loss", "perplexity"]
		tensorlist = []
		textlist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													   prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													   prefix="test")

	def print_parameters(self):
		for item in self.params:
			print('%s: %s' % (item.name, item.get_shape()))

	def step_decoder(self, session, data, forward_only=False):
		input_feed = {self.raw_input_data: data['sent'],
					  self.raw_input_data_length: data['sent_length'],
					  self.use_prior: False}
		if forward_only:
			output_feed = [self.loss,
						   self.decoder_distribution_teacher,
						   self.ppl_loss
						  ]
		else:
			output_feed = [self.loss,
						   self.gradient_norm,
						   self.update,
						   self.ppl_loss
						   ]
		return session.run(output_feed, input_feed)

	def inference(self, session, data):
		input_feed = {self.raw_input_data: data['sent'],
					  self.raw_input_data_length: data['sent_length']
					 }
		output_feed = [self.generation_index]
		return session.run(output_feed, input_feed)

	def evaluate(self, sess, data, batch_size, key_name):
		loss_step = np.zeros((1,))
		ppl_loss_step = 0
		times = 0
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			outputs = self.step_decoder(sess, batched_data, forward_only=True)
			loss_step += outputs[0]
			ppl_loss_step += outputs[-1]
			times += 1
			batched_data = data.get_next_batch(key_name)

		loss_step /= times
		ppl_loss_step /= times

		print('    loss: %.2f' % loss_step)
		return loss_step, ppl_loss_step

	def train_process(self, sess, data, args):

		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		ppl_loss_step = 0
		previous_losses = [1e18] * 5
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")

		print( "EVERY THING FINE!!")

		for epoch_step in range(args.epochs):
			while batched_data != None:
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					print("Epoch %d global step %d learning rate %.4f step-time %.2f"
						  % (epoch_step, self.global_step.eval(), self.learning_rate.eval(),
							 time_step))
					print('    loss: %.2f' % loss_step)
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps,
									  {'loss': loss_step,
									   'perplexity': np.exp(ppl_loss_step),
									   })
					#self.saver.save(sess, '%s/checkpoint_latest' % args.model_dir, global_step=self.global_step)\
					self.store_checkpoint(sess, '%s/checkpoint_latest/checkpoint' % args.model_dir, "latest")

					devout = self.evaluate(sess, data, args.batch_size, "dev")
					self.devSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': devout[0],
																					   'perplexity': np.exp(devout[1]),
					})

					testout = self.evaluate(sess, data, args.batch_size, "test")
					self.testSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': testout[0],
																						'perplexity': np.exp(
																							testout[1]),
					})

					if np.sum(loss_step) > max(previous_losses):
						sess.run(self.learning_rate_decay_op)
					if devout[0] < best_valid:
						best_valid = devout[0]
						self.store_checkpoint(sess, '%s/checkpoint_best/checkpoint' % args.model_dir, "best")

					previous_losses = previous_losses[1:] + [np.sum(loss_step)]
					loss_step, time_step = np.zeros((1,)), .0
					ppl_loss_step = 0

				start_time = time.time()
				outputs = self.step_decoder(sess, batched_data)
				loss_step += outputs[0] / args.checkpoint_steps
				ppl_loss_step += outputs[-1] / args.checkpoint_steps

				time_step += (time.time() - start_time) / args.checkpoint_steps
				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")

	def test_process(self, sess, data, args):
		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")
		results = []
		while batched_data != None:
			batched_responses_id = self.inference(sess, batched_data)[0]
			gen_prob = self.step_decoder(sess, batched_data, forward_only=True)[1]
			metric1_data = {'sent_allvocabs': np.array(batched_data['sent_allvocabs']),
							'sent_length': np.array(batched_data['sent_length']),
							'gen_log_prob': np.array(gen_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				result_token = []
				response_id_list = response_id.tolist()
				response_token = data.index_to_sen(response_id_list)
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id)+1]
				else:
					result_id = response_id_list
				for token in response_token:
					if token != data.ext_vocab[data.eos_id]:
						result_token.append(token)
					else:
						break
				results.append(result_token)
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results)}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w') as f:
			print("Test Result:")
			for key, value in res.items():
				if isinstance(value, float):
					print("\t%s:\t%f" % (key, value))
					f.write("%s:\t%f\n" % (key, value))
			for i in range(len(res['gen'])):
				f.write("%s\n" % " ".join(res['gen'][i]))

		print("result output to %s." % test_file)

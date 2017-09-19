import os, math, time
import cPickle as pickle
import tensorflow as tf
import multiprocessing
from data import DataGenerator, process_batch
from data_preprocess import TOKEN_DICT, _GO, _EOS
from create_tensorboard_start_script import generate_tensorboard_script
from utils import clear_folder, model_meta_file


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


class Seq2SeqModel(object):


    def __init__(self, sess_config, model_path, log_path, vocab_size, num_batches,
                 learning_rate=0.0005, batch_size = 32, embedding_size=64,
                 model_name='seq2seq_test', hidden_units=32, display_steps=200,
                 saving_steps=100, eval_mode=False, restore_model=False, use_raw_rnn=False,):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_batches = num_batches
        self.model_name = model_name
        self.display_steps = display_steps
        self.saving_steps = saving_steps
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.log_path = log_path
        self.USE_RAW_RNN = use_raw_rnn

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        self.global_step = 0

        if eval_mode:
            self._restore_model(training_mode=False)
            self._build_eval_graph()

        elif restore_model:
            self._restore_model()

        else:
            clear_folder(self.log_path)
            clear_folder(self.model_path)
            generate_tensorboard_script(self.log_path)  # create the script to start a tensorboard session
            self._build_graph()

    def _restore_model(self, training_mode=True):
        ''' restore model from local file, two different mode: training and evaluation
        '''
        with self.graph.as_default():
            self._restore_placeholders()
            if training_mode:
                self.saver = tf.train.import_meta_graph(model_meta_file(self.model_path))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
                self._restore_training_variables()
                print 'restore trained models from {}'.format(self.model_path)
            else:
                self._restore_eval_variables()
                print 'restore eval models from {}'.format(self.model_path)

    def _build_graph(self):
        ''' build the training graph
        '''
        with self.graph.as_default():
            self._init_placeholders()
            self._init_variable()
            self._build_encoder()
            if self.USE_RAW_RNN:
                self._build_raw_rnn_decoder()
            else:
                self._build_dynamic_rnn_decoder()
            self._build_optimizer()
            self.saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _build_eval_graph(self):
        ''' build the evaluation graph
        '''
        with self.graph.as_default():
            ## transpose the dimension of embedded input to [batch_size, max_time, embedded_size]
            embedded_inputs = tf.transpose(self.encoder_inputs_embedded, [1, 0, 2])
            self.mean_embedded_inputs = tf.reduce_mean(embedded_inputs, axis=1)
            self.max_embedded_inputs = tf.reduce_max(embedded_inputs, axis=1)
            self.min_embedded_inputs = tf.reduce_min(embedded_inputs, axis=1)

            ## change the dimension to [batch_size, max_time, cell.output_size]
            encoder_outputs_ = tf.transpose(self.encoder_outputs, [1, 0, 2])
            self.mean_encoder_outputs = tf.reduce_mean(encoder_outputs_, axis=1)
            self.max_encoder_outputs = tf.reduce_max(encoder_outputs_, axis=1)
            self.min_encoder_outputs = tf.reduce_min(encoder_outputs_, axis=1)


    def eval_by_batch(self, batch):
        ''' run the outupt tensors with a batch of input titles
        Return:
            a set of three different types of embedding outputs:
            1. the mean/max/min from the word embedding
            2. the mean/max/min from the encoder outputs
            3. the last hidden state ouput
        '''
        encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
        feed_content = { self.encoder_inputs: encoder_inputs_,
                         self.dropout_input_keep_prob: 1.}

        mean_embedded_inputs_, max_embedded_inputs_, min_embedded_inputs_ = self.sess.run([self.mean_embedded_inputs,
                                                                                           self.max_embedded_inputs,
                                                                                           self.min_embedded_inputs],
                                                                                         feed_content)
        embedded_input_sets = (mean_embedded_inputs_, max_embedded_inputs_, min_embedded_inputs_)

        mean_encoder_outputs, max_encoder_outputs, min_encoder_outputs = self.sess.run([self.mean_encoder_outputs,
                                                                                        self.max_encoder_outputs,
                                                                                        self.min_encoder_outputs],
                                                                                       feed_content)
        encode_ouput_sets = (mean_encoder_outputs, max_encoder_outputs, min_encoder_outputs)

        final_cell_state_, final_hidden_state_ = self.sess.run([self.final_cell_state, self.final_hidden_state], feed_content)
        hidden_state_sets = (final_cell_state_, final_hidden_state_)

        return embedded_input_sets, encode_ouput_sets, hidden_state_sets


    def _init_placeholders(self):
        '''follow the example and use the time-major
        '''
        with tf.name_scope('initial_inputs'):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
            self.dropout_input_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_input_keep_prob')
            self.decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')

    def _init_variable(self):
        # Initialize embeddings to have variance=1, encoder and decoder share the same embeddings
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
        self.embeddings = tf.get_variable(name='embedding_matrix',
                                          shape=[self.vocab_size, self.embedding_size],
                                          initializer=initializer,
                                          dtype=tf.float32)

        self.projection_weights = tf.Variable(tf.random_uniform([self.hidden_units, self.vocab_size], -1, 1),
                                              dtype=tf.float32,
                                              name='projection_weights')

        self.projection_bias = tf.Variable(tf.zeros([self.vocab_size]),
                                           dtype=tf.float32,
                                           name='projection_bias')

        tf.summary.histogram('{}_histogram'.format('embeddings'), self.embeddings)
        tf.summary.histogram('{}_histogram'.format('projection_weights'), self.projection_weights)
        tf.summary.histogram('{}_histogram'.format('projection_bias'), self.projection_bias)

        self.global_saving_steps = tf.Variable(0, name='global_saving_steps', trainable=False, dtype=tf.int32)
        self.increment_saving_step_op = tf.assign(self.global_saving_steps,
                                                  self.global_saving_steps + self.saving_steps,
                                                  name="increment_saving_step_op")

    def _build_encoder(self):
        with tf.name_scope('encoder'):
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs, name="encoder_inputs_embedded")
            encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=self.dropout_input_keep_prob)
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell,
                self.encoder_inputs_embedded,
                dtype=tf.float32,
                time_major=True,
                scope="dynamic_encoder")
            self.final_cell_state = self.encoder_final_state[0]
            self.final_hidden_state = self.encoder_final_state[1]
            print "self.encoder_outputs: ", self.encoder_outputs
            print "self.final_cell_state: ", self.final_cell_state
            print "self.final_hidden_state: ", self.final_hidden_state

    def _build_raw_rnn_decoder(self):
        with tf.name_scope('raw_rnn_decoder'):
            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            #decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.dropout_input_keep_prob)

            ## give three extra space for error
            decoder_lengths = self.decoder_inputs_length  ## consider the first <_GO>
            ## create the embedded <GO>
            assert TOKEN_DICT[_GO] == 1
            go_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='_GO')
            go_step_embedded = tf.nn.embedding_lookup(self.embeddings, go_time_slice)

            def loop_fn_initial():
                '''returns the expected sets of outputs for the initial LSTM unit.
                the external variable `encoder_final_state` is used as initial_cell_state
                '''
                initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                initial_input = go_step_embedded
                initial_cell_state = self.encoder_final_state
                initial_cell_output = None
                initial_loop_state = None  # we don't need to pass any additional information
                return (initial_elements_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)

            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
                '''create the outputs for next LSTM unit
                A projection with word embedding matrix is used to find the next input, instead of
                using the target se in `dynamic_rnn`.
                '''
                def get_next_input():
                    output_logits = tf.add(tf.matmul(previous_output, self.projection_weights), self.projection_bias)
                    prediction = tf.argmax(output_logits, axis=1)
                    next_input = tf.nn.embedding_lookup(self.embeddings, prediction)
                    return next_input

                elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
                # defining if corresponding sequence has ended
                cur_input = get_next_input()
                cur_state = previous_state
                cur_output = previous_output
                loop_state = None
                return (elements_finished, cur_input, cur_state, cur_output, loop_state)

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:  # time == 0
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

            decoder_outputs_tensor_array, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
            self.decoder_outputs = decoder_outputs_tensor_array.stack()



    def _build_dynamic_rnn_decoder(self):
        with tf.name_scope('dynamic_rnn_decoder'):
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.dropout_input_keep_prob)
            self.decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                decoder_cell,
                decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32,
                time_major=True,
                scope="decoder")


    def _build_optimizer(self):

        # project the every hidden output from LSTM unit output to the word embedding matrix
        with tf.name_scope('outputs_projection'):
            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))
            decoder_outputs_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
            decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.projection_weights), self.projection_bias)
            decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.vocab_size))
            self.decoder_prediction = tf.argmax(decoder_logits, 2, name='decoder_prediction')
            tf.summary.histogram('{}_histogram'.format('decoder_prediction'), self.decoder_prediction)

        with tf.name_scope('objective_function'):
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits=decoder_logits)
            self.loss = tf.reduce_mean(stepwise_cross_entropy, name='loss')
            self.single_variable_summary(self.loss, 'loss')

        with tf.name_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name='train_op')



    def next_feed(self, batches, dropout_input_keep_prob):
        if self.USE_RAW_RNN:
            training_batch, target_batch = next(batches)
            encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in training_batch])
            decoder_targets_, decode_sequence_lengths_ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in target_batch])
            return {self.encoder_inputs: encoder_inputs_,
                    self.decoder_targets: decoder_targets_,
                    self.decoder_inputs_length: decode_sequence_lengths_,
                    self.dropout_input_keep_prob: dropout_input_keep_prob}
        else:
            batch = next(batches)
            encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
            decoder_inputs_, _ = process_batch([[TOKEN_DICT[_GO]] + sequence for sequence in batch])
            decoder_targets_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
            return {self.encoder_inputs: encoder_inputs_,
                    self.decoder_inputs: decoder_inputs_,
                    self.decoder_targets: decoder_targets_,
                    self.dropout_input_keep_prob : dropout_input_keep_prob}

    @staticmethod
    def single_variable_summary(var, name):
        reduce_mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
        tf.summary.histogram('{}_histogram'.format(name), var)

    def _restore_placeholders(self):
        self.encoder_inputs = self.sess.graph.get_tensor_by_name("initial_inputs/encoder_inputs:0")
        self.decoder_inputs = self.sess.graph.get_tensor_by_name("initial_inputs/decoder_inputs:0")
        self.decoder_targets = self.sess.graph.get_tensor_by_name("initial_inputs/decoder_targets:0")
        self.decoder_inputs_length = self.sess.graph.get_tensor_by_name("initial_inputs/decoder_inputs_length:0")
        self.dropout_input_keep_prob = self.sess.graph.get_tensor_by_name("initial_inputs/dropout_input_keep_prob:0")


    def _restore_eval_variables(self):
        self.encoder_inputs_embedded = self.sess.graph.get_tensor_by_name("encoder/encoder_inputs_embedded:0")
        self.encoder_outputs = self.sess.graph.get_tensor_by_name("encoder/rnn/TensorArrayStack/TensorArrayGatherV3:0")
        self.final_cell_state = self.sess.graph.get_tensor_by_name("encoder/rnn/while/Exit_2:0")
        self.final_hidden_state = self.sess.graph.get_tensor_by_name("encoder/rnn/while/Exit_3:0")


    def _restore_training_variables(self):
        self.global_saving_steps = self.sess.graph.get_tensor_by_name("global_saving_steps:0")
        self.increment_saving_step_op = self.sess.graph.get_tensor_by_name("increment_saving_step_op:0")
        self.train_op = self.sess.graph.get_operation_by_name("optimizer/train_op")
        self.loss = self.sess.graph.get_tensor_by_name("objective_function/loss:0")
        self.decoder_prediction = self.sess.graph.get_tensor_by_name("outputs_projection/decoder_prediction:0")
        self.global_step = self.sess.run(self.global_saving_steps)

    def _saving_step_run(self):
        _ = self.sess.run(self.increment_saving_step_op)
        self.saver.save(self.sess, os.path.join(self.model_path, 'models'), global_step=self.global_step)


    def _display_step_run(self, start_time, feed_content, reverse_token_dict):
        summary, loss_value = self.sess.run([self.merged_summary_op, self.loss], feed_content)
        print 'step {}, minibatch loss: {}'.format(self.global_step, loss_value)
        self.writer.add_summary(summary, self.global_step)
        if self.global_step != 1:
            print 'every {} steps, it takes {:.2f} minutes...'.format(self.display_steps, (1.*time.time()-start_time)/60.)
        predict_ = self.sess.run(self.decoder_prediction, feed_content)
        for i, (inp, pred) in enumerate(zip(feed_content[self.encoder_inputs].T, predict_.T)):
            print '  sample {}:'.format(i + 1)
            print '    input     > {}'.format(map(reverse_token_dict.get, inp))
            print '    predicted > {}'.format(map(reverse_token_dict.get, pred))
            if i >= 5:
                break
        return time.time()


    def train(self, batches, reverse_token_dict, dropout_input_keep_prob=0.8):
        with self.graph.as_default():
            self.writer = tf.summary.FileWriter(self.log_path, graph=self.graph)
            self.merged_summary_op = tf.summary.merge_all()
            start_time = time.time()
            while self.global_step < self.num_batches:
                feed_content = self.next_feed(batches, dropout_input_keep_prob)
                _ = self.sess.run([self.train_op], feed_content)
                self.global_step += 1

                if self.global_step % self.saving_steps == 0:
                    self._saving_step_run()

                if self.global_step == 1 or self.global_step % self.display_steps == 0:
                    start_time = self._display_step_run(start_time, feed_content, reverse_token_dict)

            self.saver.save(self.sess, os.path.join(self.model_path, 'final_model'), global_step=self.global_step)


def retrieve_reverse_token_dict(picke_file_path, key='reverse_token_dict'):
    with open(picke_file_path, 'rb') as raw_input:
        content = pickle.load(raw_input)
    return content[key]


def model_train():

    #pickle_file = 'processed_titles_data.pkl'
    pickle_file = 'scramble_titles_data.pkl'

    epoch_num = 2000
    batch_size = 16
    USE_RAW_RNN = True
    USE_GPU = False

    # PAD = 0 ## default padding is 0
    NUM_THREADS = 2 * multiprocessing.cpu_count() - 1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path, dual_outputs=USE_RAW_RNN)
    batches = dataGen.generate_sequence(batch_size)

    model_config = {}
    model_config['restore_model'] = False
    model_config['eval_mode'] = False
    model_config['learning_rate'] = 0.00001
    model_config['model_name'] = 'seq2seq_model'
    model_config['batch_size'] = batch_size
    model_config['use_raw_rnn'] = USE_RAW_RNN
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['num_batches'] = int(dataGen.data_size*epoch_num/model_config['batch_size'])

    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])

    if USE_GPU:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    reverse_token_dict = retrieve_reverse_token_dict(pickle_file_path)
    print 'total #batches: {}, vocab_size: {}'.format(model_config['num_batches'], model_config['vocab_size'])

    model = Seq2SeqModel(**model_config)
    model.train(batches, reverse_token_dict, dropout_input_keep_prob=0.5)


def model_predict():
    # pickle_file = 'processed_titles_data.pkl'
    pickle_file = 'scramble_titles_data.pkl'

    epoch_num = 2000
    batch_size = 16
    USE_RAW_RNN = True
    USE_GPU = False

    # PAD = 0 ## default padding is 0
    NUM_THREADS = 2 * multiprocessing.cpu_count() - 1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path, dual_outputs=USE_RAW_RNN)
    batches = dataGen.generate_sequence(batch_size)

    model_config = {}
    model_config['restore_model'] = False
    model_config['eval_mode'] = False
    model_config['learning_rate'] = 0.00001
    model_config['model_name'] = 'seq2seq_model'
    model_config['batch_size'] = batch_size
    model_config['use_raw_rnn'] = USE_RAW_RNN
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['num_batches'] = int(dataGen.data_size * epoch_num / model_config['batch_size'])

    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])

    if USE_GPU:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    batch = next(batches)
    model = Seq2SeqModel(**model_config)
    embedded_input_sets, encode_ouput_sets, hidden_state_sets = model.eval_by_batch(batch)
    print embedded_input_sets[0]
    print embedded_input_sets[1]
    print embedded_input_sets[2]

if __name__ == '__main__':
    model_train()
    #model_predict()
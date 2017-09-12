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

    # PAD = 0 ## default padding is 0
    NUM_THREADS = 2 * multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    def __init__(self, batches, vocab_size, num_batches, model_name='seq2seq_test',
                 embedding_size=64, hidden_units=32, display_steps=200, saving_step_rate=5, use_gpu=False):

        self.batches = batches
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.num_batches = num_batches
        self.model_name = model_name
        self.display_steps = display_steps
        self.saving_steps = saving_step_rate * self.display_steps

        self.model_path = create_local_model_path(self.COMMON_PATH, self.model_name)
        self.log_path = create_local_log_path(self.COMMON_PATH, self.model_name)
        generate_tensorboard_script(self.log_path)  # create the script to start a tensorboard session

        if use_gpu:
            self.config = tf.ConfigProto(log_device_placement=False,
                                         gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
            self.config = tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)

    def _init_placeholders(self):
        '''follow the example and use the time-major
        '''
        with tf.name_scope('initial_inputs'):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
            self.dropout_input_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_input_keep_prob')
            self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

        with tf.name_scope('word_embedding'):
            # Initialize embeddings to have variance=1, encoder and decoder share the same embeddings
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            #self.embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
            embeddings = tf.get_variable(name='embedding_matrix',
                                         shape=[self.vocab_size, self.embedding_size],
                                         initializer=initializer,
                                         dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

    def _build_sequence(self):
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)
        with tf.name_scope('encoder_decoder_sequence'):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=self.dropout_input_keep_prob)
            encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell,
                self.encoder_inputs_embedded,
                dtype=tf.float32,
                time_major=True,
                scope='encoder')

            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.dropout_input_keep_prob)
            self.decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                decoder_cell,
                self.decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32,
                time_major=True,
                scope="decoder")

    def _build_optimizer(self, learning_rate=0.0001):
        with tf.name_scope('decoder_projection'):
            # project the decoder output
            decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
            self.decoder_prediction = tf.argmax(decoder_logits, 2)
            # histogram the decoder_prediction
            tf.summary.histogram('{}_histogram'.format('decoder_prediction'), self.decoder_prediction)

        with tf.name_scope('objective_function'):
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits=decoder_logits)
            self.loss = tf.reduce_mean(stepwise_cross_entropy)
            self.single_variable_summary(self.loss, 'objective_func_loss')

        with tf.name_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def next_feed(self, dropout_input_keep_prob):
        batch = next(self.batches)
        encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
        decoder_targets_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
        decoder_inputs_, _ = process_batch([[TOKEN_DICT[_GO]] + sequence for sequence in batch])
        return {
            self.encoder_inputs: encoder_inputs_,
            self.decoder_inputs: decoder_inputs_,
            self.decoder_targets: decoder_targets_,
            self.dropout_input_keep_prob : dropout_input_keep_prob
        }

    @staticmethod
    def single_variable_summary(var, name):
        reduce_mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
        tf.summary.histogram('{}_histogram'.format(name), var)

    def _restore_placeholders(self, sess):
        self.encoder_inputs = sess.graph.get_tensor_by_name("initial_inputs/encoder_inputs:0")
        self.decoder_inputs = sess.graph.get_tensor_by_name("initial_inputs/decoder_inputs:0")
        self.decoder_targets = sess.graph.get_tensor_by_name("initial_inputs/decoder_targets:0")
        self.dropout_input_keep_prob = sess.graph.get_tensor_by_name("initial_inputs/dropout_input_keep_prob:0")

    def _restore_operation_variables(self, sess):
        self.global_step = sess.graph.get_tensor_by_name("initial_inputs/global_step:0")
        self.increment_global_step_op = sess.graph.get_tensor_by_name("Assign:0")
        self.train_op = sess.graph.get_operation_by_name("optimizer/Adam")
        self.loss = sess.graph.get_tensor_by_name("objective_function/Mean:0")
        self.decoder_prediction = sess.graph.get_tensor_by_name("decoder_projection/ArgMax:0")

    def train(self, learning_rate, reverse_token_dict, dropout_input_keep_prob=0.8, restore_model=False):

        if not restore_model:
            clear_folder(self.log_path)
            clear_folder(self.model_path)

            self._init_placeholders()
            self._build_sequence()
            self._build_optimizer(learning_rate)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
        else:
            saver = tf.train.import_meta_graph(model_meta_file(self.model_path))

        writer = tf.summary.FileWriter(self.log_path)
        merged_summary_op = tf.summary.merge_all()
       
        with tf.Session(config=self.config) as sess:
            if not restore_model:
                sess.run(init)
                writer.add_graph(sess.graph)
            else:
                print 'restore trained models from {}'.format(self.model_path)
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                self._restore_placeholders(sess)
                self._restore_operation_variables(sess)

            step = 0
            start_time = time.time()
            while step < self.num_batches:
                feed_content = self.next_feed(dropout_input_keep_prob)
                _, step, summary, loss_value = sess.run([self.train_op,
                                                         self.increment_global_step_op,
                                                         merged_summary_op,
                                                         self.loss], feed_content)

                if step % self.saving_steps == 0:
                    saver.save(sess, os.path.join(self.model_path, 'models'), global_step=step)

                if step == 1 or step % self.display_steps == 0:
                    print 'step {}, minibatch loss: {}'.format(step, loss_value)
                    writer.add_summary(summary, step)
                    if step != 1:
                        print 'every {} steps, it takes {:.2f} minutes...'.format(self.display_steps,
                                                                                  (1.*time.time()-start_time) / 60.)
                    start_time = time.time()
                    predict_ = sess.run(self.decoder_prediction, feed_content)
                    for i, (inp, pred) in enumerate(zip(feed_content[self.encoder_inputs].T, predict_.T)):
                        print '  sample {}:'.format(i + 1)
                        print '    input     > {}'.format(map(reverse_token_dict.get, inp))
                        print '    predicted > {}'.format(map(reverse_token_dict.get, pred))
                        if i >= 5:
                            break
                step += 1
            saver.save(sess, os.path.join(self.model_path, 'final_model'), global_step=step)


def retrieve_reverse_token_dict(picke_file_path, key='reverse_token_dict'):
    with open(picke_file_path, 'rb') as raw_input:
        content = pickle.load(raw_input)
    return content[key]


def main():

    pickle_file = 'processed_titles_data.pkl'
    batch_size = 16

    epoch_num = 2000
    learning_rate = 0.00001

    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path)
    batches = dataGen.generate_sequence(batch_size)

    reverse_token_dict = retrieve_reverse_token_dict(pickle_file_path)
    vocab_size = dataGen.vocab_size + 1
    num_batches = int(dataGen.data_size * epoch_num / batch_size)
    print 'total #batches: {}, vocab_size: {}'.format(num_batches, vocab_size)
    model = Seq2SeqModel(batches, vocab_size=vocab_size, num_batches=num_batches, embedding_size=32, hidden_units=256, display_steps=10000, use_gpu=True, model_name='large_32embed_256hid')
    model.train(learning_rate, reverse_token_dict, restore_model=True)

if __name__ == '__main__':
    main()

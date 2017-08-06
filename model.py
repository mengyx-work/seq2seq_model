import os, math, time
import numpy as np
import tensorflow as tf
import multiprocessing
from data import DataGenerator, process_batch
from create_tensorboard_start_script import generate_tensorboard_script
from utils import clear_folder


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


class Seq2SeqModel(object):

    PAD = 0
    EOS = 1

    NUM_THREADS = 2 * multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    def __init__(self, batches, vocab_size, num_batches, model_name='seq2seq_test', embedding_size=32, hidden_units=16, display_steps=500, use_gpu=False):

        self.batches = batches
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.num_batches = num_batches
        self.model_name = model_name
        self.display_steps = display_steps

        self.model_path = create_local_model_path(self.COMMON_PATH, self.model_name)
        self.log_path = create_local_log_path(self.COMMON_PATH, self.model_name)
        generate_tensorboard_script(self.log_path)  # create the script to start a tensorboard session

        if use_gpu:
            self.config = tf.ConfigProto(log_device_placement=False,
                                         gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))
        else:
            self.config = tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)


    def _init_placeholders(self):
        '''follow the example and use the time-major

        '''
        with tf.name_scope('initial_inputs'):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
            #self.embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

        with tf.name_scope('word_embedding'):
            # Initialize embeddings to have variance=1, encoder and decoder share the same embeddings
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            embeddings = tf.get_variable(name='embedding',
                                         shape=[self.vocab_size, self.embedding_size],
                                         initializer=initializer,
                                         dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

    def _build_sequence(self):
        with tf.name_scope('encoder_decoder_sequence'):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell,
                self.encoder_inputs_embedded,
                dtype=tf.float32,
                time_major=True,
                scope='decoder')

            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            self.decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                decoder_cell,
                self.decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32,
                time_major=True,
                scope="plain_decoder")

    def _build_optimizer(self, learning_rate=0.0001):
        with tf.name_scope('convert_decoder_output'):
            decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)  # project the decoder output
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

    def next_feed(self):
        batch = next(self.batches)
        encoder_inputs_, _ = process_batch(batch)
        decoder_targets_, _ = process_batch([(sequence) + [self.EOS] for sequence in batch])
        decoder_inputs_, _ = process_batch([[self.EOS] + (sequence) for sequence in batch])
        return {
            self.encoder_inputs: encoder_inputs_,
            self.decoder_inputs: decoder_inputs_,
            self.decoder_targets: decoder_targets_,
        }

    @staticmethod
    def single_variable_summary(var, name):
        reduce_mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
        tf.summary.histogram('{}_histogram'.format(name), var)

    def train(self, learning_rate):
        clear_folder(self.log_path)
        clear_folder(self.model_path)

        self._init_placeholders()
        self._build_sequence()
        self._build_optimizer(learning_rate)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        writer = tf.summary.FileWriter(self.log_path)
        merged_summary_op = tf.summary.merge_all()
        max_batches = 6001
        start_time = time.time()
        with tf.Session(config=self.config) as sess:
            sess.run(init)
            writer.add_graph(sess.graph)
            step = 0
            while step < self.num_batches:
                fd = self.next_feed()
                _, summary, loss_value = sess.run([self.train_op, merged_summary_op, self.loss], fd)
                if step == 0 or step % self.display_steps == 0:
                    writer.add_summary(summary, step)
                    saver.save(sess, os.path.join(self.model_path, 'models'), global_step=step)
                    print 'step {}, minibatch loss: {}, taking {:.2f} minutes'.format(step,
                                                                                       loss_value,
                                                                                       (1.*time.time()-start_time) / 60.)
                    predict_ = sess.run(self.decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[self.encoder_inputs].T, predict_.T)):
                        print '  sample {}:'.format(i + 1)
                        print '    input     > {}'.format(inp)
                        print '    predicted > {}'.format(pred)
                        if i >= 5:
                            break
                step += 1

            saver.save(sess, os.path.join(self.model_path, 'final_model'), global_step=step)

def main():
    pickle_file = '~/content.pkl'
    batch_size = 10
    epoch_num = 10
    learning_rate = 0.00001
    dataGen = DataGenerator(pickle_file)
    batches = dataGen.generate_sequence(batch_size)
    vocab_size = dataGen.vocab_size + 1
    num_batches = int(dataGen.data_size * epoch_num / batch_size)
    model = Seq2SeqModel(batches, vocab_size=vocab_size, num_batches=num_batches)
    model.train(learning_rate)

if __name__ == '__main__':
    main()

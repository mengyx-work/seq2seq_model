import os, math, time
import cPickle as pickle
import tensorflow as tf
import multiprocessing
from data import DataGenerator, process_batch
from data_preprocess import TOKEN_DICT, _GO, _PAD, _EOS
from create_tensorboard_start_script import generate_tensorboard_script
from utils import clear_folder, model_meta_file


def _embed_variables_by_bedding_matrix(placeholder, embedding_name, config, re_use=True):
    '''to embed any inputs (placeholder/variables) of word index by the word embedding matrix
    Args:
        placeholder (TF placeholder/variables): the variable to embed, expect integer index
        embedding_name (string): the name of embedded variable
        config (config object): the configure
    Return:
        inputs_embedded (TF variable): embedded variables
    '''
    if not re_use:
        with tf.variable_scope('word_embedding'):
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            embeddings = tf.get_variable(name='word_embedding_matrix',
                                        shape=[config.vocab_size, config.embedding_size],
                                        initializer=initializer,
                                        dtype=tf.float32)
    else:
        with tf.variable_scope('word_embedding', reuse=True):
            embeddings = tf.get_variable('word_embedding_matrix')
    inputs_embedded = tf.nn.embedding_lookup(embeddings, placeholder, name=embedding_name)
    return inputs_embedded



def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


def _init_placeholders():
    '''follow the example and use the time-major
    '''
    placeholders = {}
    with tf.name_scope("initial_inputs"):
        placeholders['decoder_inputs'] = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        placeholders['encoder_inputs'] = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        placeholders['decoder_targets'] = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        placeholders['decoder_inputs_length'] = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')
        placeholders['dropout_input_keep_prob'] = tf.placeholder(dtype=tf.float32, name='dropout_input_keep_prob')
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    return placeholders, global_step


def _restore_placeholders(sess):
    placeholders = {}
    placeholders['decoder_inputs'] = sess.graph.get_tensor_by_name("initial_inputs/encoder_inputs:0")
    placeholders['encoder_inputs'] = sess.graph.get_tensor_by_name("initial_inputs/decoder_inputs:0")
    placeholders['decoder_targets'] = sess.graph.get_tensor_by_name("initial_inputs/decoder_targets:0")
    placeholders['dropout_input_keep_prob'] = sess.graph.get_tensor_by_name("initial_inputs/dropout_input_keep_prob:0")
    placeholders['decoder_inputs_length'] = sess.graph.get_tensor_by_name("initial_inputs/decoder_inputs_length:0")
    global_step = sess.graph.get_tensor_by_name("initial_inputs/global_step:0")
    return placeholders, global_step


def _restore_variables_for_inference(sess):
    mean_encoder_inputs_embedded_ = sess.graph.get_tensor_by_name("inference/Mean:0")
    mean_encoder_outputs_ = sess.graph.get_tensor_by_name("inference/Mean_1:0")
    final_cell_state_ = sess.graph.get_tensor_by_name("encoder_sequence/encoder/while/Exit_2:0")
    final_hidden_state_ = sess.graph.get_tensor_by_name("encoder_sequence/encoder/while/Exit_3:0")
    return mean_encoder_inputs_embedded_, mean_encoder_outputs_, final_cell_state_, final_hidden_state_


def _build_sequence(placeholders, config):
    '''core of the sequence model.
    '''

    with tf.name_scope('sequence_variables'):
        # Initialize embeddings to have variance=1, encoder and decoder share the same embeddings
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
        embeddings = tf.get_variable(name='word_embedding_matrix',
                                     shape=[config.vocab_size, config.embedding_size],
                                     initializer=initializer,
                                     dtype=tf.float32)

        projection_weights = tf.Variable(tf.random_uniform([config.hidden_units, config.vocab_size], -1, 1),
                                         dtype=tf.float32,
                                         name='projection_weights')

        projection_bias = tf.Variable(tf.zeros([config.vocab_size]),
                                      dtype=tf.float32,
                                      name='projection_bias')

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,
                                                         placeholders['encoder_inputs'],
                                                         name='encoder_inputs_embedded')

    with tf.name_scope('encoder_sequence'):
        encoder_cell = tf.contrib.rnn.LSTMCell(config.hidden_units)
        encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=placeholders['dropout_input_keep_prob'])
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell,
            encoder_inputs_embedded,
            dtype=tf.float32,
            time_major=True,
            scope='encoder')

    with tf.name_scope('inference'):
        ## transpose the dimension of embedded input to [batch_size, max_time, embedded_size]
        encoder_inputs_embedded_ = tf.transpose(encoder_inputs_embedded, [1, 0, 2])
        mean_encoder_inputs_embedded = tf.reduce_mean(encoder_inputs_embedded_, axis=1)

        ## change the dimension to [batch_size, max_time, cell.output_size]
        encoder_outputs_ = tf.transpose(encoder_outputs, [1, 0, 2])
        mean_encoder_outputs = tf.reduce_mean(encoder_outputs_, axis=1)

        final_cell_state = encoder_final_state[0]
        final_hidden_state = encoder_final_state[1]


    with tf.name_scope('decoder_sequence'):
        decoder_cell = tf.contrib.rnn.LSTMCell(config.hidden_units)
        ## give three extra space for error
        decoder_lengths = placeholders['decoder_inputs_length'] + 1   ## consider the first <_GO>
        ## create the embedded _GO
        assert TOKEN_DICT[_GO] == 1
        go_time_slice = tf.ones([config.batch_size], dtype=tf.int32, name='EOS')
        go_step_embedded = tf.nn.embedding_lookup(embeddings, go_time_slice)

        def loop_fn_initial():
            '''returns the expected sets of outputs for the initial LSTM unit.
            the external variable `encoder_final_state` is used as initial_cell_state
            '''
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = go_step_embedded
            initial_cell_state = encoder_final_state
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
                output_logits = tf.add(tf.matmul(previous_output, projection_weights), projection_bias)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(embeddings, prediction)
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
        decoder_outputs = decoder_outputs_tensor_array.stack()

    with tf.name_scope('outputs_projection'):
        ## project the last hidden output from LSTM unit outputs to the word matrix
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, projection_weights), projection_bias)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, config.vocab_size))
        decoder_prediction = tf.argmax(decoder_logits, 2)
    tf.summary.histogram('{}_histogram'.format('decoder_prediction'), decoder_prediction)

    inference_set = (mean_encoder_inputs_embedded, mean_encoder_outputs, final_cell_state, final_hidden_state)
    return decoder_prediction, decoder_logits, inference_set


def _build_optimizer(placeholders, decoder_logits, config):
    with tf.name_scope('objective_function'):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(placeholders['decoder_targets'], depth=config.vocab_size, dtype=tf.float32),
                logits=decoder_logits)
        loss = tf.reduce_mean(stepwise_cross_entropy)
        single_variable_summary(loss, 'objective_func_loss')

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
    return loss, train_op


def single_variable_summary(var, name):
    reduce_mean = tf.reduce_mean(var)
    tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
    tf.summary.histogram('{}_histogram'.format(name), var)


def _restore_operation_variables(sess):
    train_op = sess.graph.get_operation_by_name("optimizer/Adam")
    loss = sess.graph.get_tensor_by_name("objective_function/Mean:0")
    decoder_prediction = sess.graph.get_tensor_by_name("outputs_projection/ArgMax:0")
    return decoder_prediction, loss, train_op


def dual_next_feed(placeholders, batches, dropout_input_keep_prob):
    training_batch, target_batch = next(batches)
    encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in training_batch])
    decoder_targets_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in target_batch])
    ## the processing step for `dyanmic_rnn`
    #decoder_inputs_, _ = process_batch([[TOKEN_DICT[_GO]] + sequence for sequence in decoder_targets_])

    ## the processing step for `raw_rnn`
    decoder_inputs_, decode_sequence_lengths_ = process_batch([sequence for sequence in target_batch])
    return {
        placeholders['encoder_inputs']: encoder_inputs_,
        placeholders['decoder_inputs']: decoder_inputs_,
        placeholders['decoder_targets']: decoder_targets_,
        placeholders['decoder_inputs_length'] : decode_sequence_lengths_,
        placeholders['dropout_input_keep_prob'] : dropout_input_keep_prob
    }


def next_feed(placeholders, batches, dropout_input_keep_prob):
    batch = next(batches)
    encoder_inputs_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
    decoder_targets_, _ = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
    ## the processing step for `dyanmic_rnn`
    #decoder_inputs_, _ = process_batch([[TOKEN_DICT[_GO]] + sequence for sequence in batch])

    ## the processing step for `raw_rnn`
    decoder_inputs_, decode_sequence_lengths_ = process_batch([sequence for sequence in batch])
    return {
        placeholders['encoder_inputs']: encoder_inputs_,
        placeholders['decoder_inputs']: decoder_inputs_,
        placeholders['decoder_targets']: decoder_targets_,
        placeholders['decoder_inputs_length'] : decode_sequence_lengths_,
        placeholders['dropout_input_keep_prob'] : dropout_input_keep_prob
    }


def train(config, batches, reverse_token_dict, dropout_input_keep_prob=0.8, restore_model=False, dual_outputs=False):

    if not restore_model:
        clear_folder(config.log_path)
        clear_folder(config.model_path)

        placeholders, global_step_ = _init_placeholders()
        increment_global_step_op = tf.assign(global_step_, global_step_ + config.batch_size, name='increment_step')
        decoder_prediction, decoder_logits, inference_set = _build_sequence(placeholders, config)
        mean_encoder_inputs_embedded_, mean_encoder_outputs_, final_cell_state_, final_hidden_state_ = inference_set
        print "final_cell_state_: ", final_cell_state_
        print "final_hidden_state_: ", final_hidden_state_
        print "mean_encoder_inputs_embedded_: ", mean_encoder_inputs_embedded_
        print "mean_encoder_outputs_: ", mean_encoder_outputs_

        loss, train_op = _build_optimizer(placeholders, decoder_logits, config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
    else:
        saver = tf.train.import_meta_graph(model_meta_file(config.model_path))

    writer = tf.summary.FileWriter(config.log_path)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session(config=config.sess_config) as sess:
        if not restore_model:
            sess.run(init)
            writer.add_graph(sess.graph)
            step = 0
        else:
            print 'restore trained models from {}'.format(config.model_path)
            increment_global_step_op = sess.graph.get_tensor_by_name("increment_step:0")
            saver.restore(sess, tf.train.latest_checkpoint(config.model_path))
            placeholders, global_step_ = _restore_placeholders(sess)
            decoder_prediction, loss, train_op = _restore_operation_variables(sess)
            step = sess.run(global_step_)  ## retrieve the step from variable

        start_time = time.time()
        while step < config.num_batches:
            if dual_outputs:
                feed_content = dual_next_feed(placeholders, batches, dropout_input_keep_prob)
            else:
                feed_content = next_feed(placeholders, batches, dropout_input_keep_prob)
            _ = sess.run([train_op], feed_content)
            step += 1

            if step == 1 or step % config.display_steps == 0:
                _, summary, loss_value = sess.run([increment_global_step_op, merged_summary_op, loss], feed_content)
                print 'step {}, minibatch loss: {}'.format(step, loss_value)
                writer.add_summary(summary, step)
                if step != 1:
                    print 'every {} steps, it takes {:.2f} minutes...'.format(config.display_steps,
                                                                              (1. * time.time() - start_time) / 60.)
                start_time = time.time()
                predict_ = sess.run(decoder_prediction, feed_content)
                for i, (inp, pred) in enumerate(zip(feed_content[placeholders['encoder_inputs']].T, predict_.T)):
                    print '  sample {}:'.format(i + 1)
                    print '  input     > {}'.format(map(reverse_token_dict.get, inp))
                    print '  predicted > {}'.format(map(reverse_token_dict.get, pred))
                    if i >= 5:
                        break

            if step % config.saving_steps == 0:
                saver.save(sess, os.path.join(config.model_path, 'models'), global_step=step)

        saver.save(sess, os.path.join(config.model_path, 'final_model'), global_step=step)


def retrieve_reverse_token_dict(picke_file_path, key='reverse_token_dict'):
    with open(picke_file_path, 'rb') as raw_input:
        content = pickle.load(raw_input)
    return content[key]


def main():

    NUM_THREADS = multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    class SequenceModelConfig():
        pass
    model_config = SequenceModelConfig()

    model_config.batch_size = 32
    model_config.epoch_num = 4000
    model_config.learning_rate = 0.0005

    model_config.embedding_size = 128
    model_config.hidden_units = 64
    model_config.display_steps = 100
    model_config.saving_steps = 1 * model_config.display_steps

    model_name = 'sequence_model_non_scrambled_data'
    model_config.model_path = create_local_model_path(COMMON_PATH, model_name)
    model_config.log_path = create_local_log_path(COMMON_PATH, model_name)
    generate_tensorboard_script(model_config.log_path)  # create the script to start a tensorboard session

    use_gpu = False
    if use_gpu:
        model_config.sess_config = tf.ConfigProto(log_device_placement=False,
                                                  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config.sess_config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    ## create the generator for data
    pickle_file = 'processed_titles_data.pkl'
    #pickle_file = 'scramble_titles_data.pkl'
    dual_outputs_ = False
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path, dual_outputs=dual_outputs_)
    batches = dataGen.generate_sequence(model_config.batch_size)

    reverse_token_dict = retrieve_reverse_token_dict(pickle_file_path)
    model_config.vocab_size = dataGen.vocab_size + 1
    model_config.num_batches = int(dataGen.data_size * model_config.epoch_num / model_config.batch_size)
    print 'total #batches: {}, vocab_size: {}'.format(model_config.num_batches, model_config.vocab_size)
    train(model_config,
          batches,
          reverse_token_dict,
          dropout_input_keep_prob=0.6,
          restore_model=False,
          dual_outputs=dual_outputs_)

if __name__ == '__main__':
    main()

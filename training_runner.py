import tensorflow as tf
import os, multiprocessing
from data import DataGenerator
from graph_model import Seq2SeqModel
from utils import create_local_model_path, create_local_log_path, retrieve_reverse_token_dict

def model_train():

    #pickle_file = 'processed_titles_data.pkl'
    pickle_file = 'scramble_titles_data.pkl'

    epoch_num = 4000
    batch_size = 32
    USE_RAW_RNN = True
    USE_GPU = True

    # PAD = 0 ## default padding is 0
    NUM_THREADS = 2 * multiprocessing.cpu_count() - 1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path, dual_outputs=USE_RAW_RNN)
    batches = dataGen.generate_sequence(batch_size)

    model_config = {}
    model_config['restore_model'] = False
    model_config['eval_mode'] = False
    model_config['learning_rate'] = 0.0005
    model_config['display_steps'] = 10000
    model_config['saving_steps'] = 20000
    model_config['embedding_size'] = 128
    model_config['hidden_units'] = 64

    #model_config['model_name'] = 'seq2seq_raw_rnn_scrambled_lemmatized_content'
    model_config['model_name'] = 'seq2seq_model'
    model_config['batch_size'] = batch_size
    model_config['use_raw_rnn'] = USE_RAW_RNN
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['num_batches'] = int(dataGen.data_size*epoch_num/model_config['batch_size'])

    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])

    if USE_GPU:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    reverse_token_dict = retrieve_reverse_token_dict(pickle_file_path)
    print 'total #batches: {}, vocab_size: {}'.format(model_config['num_batches'], model_config['vocab_size'])

    model = Seq2SeqModel(**model_config)
    model.train(batches, reverse_token_dict, dropout_input_keep_prob=0.5)


if __name__ == '__main__':
    model_train()
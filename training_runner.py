import tensorflow as tf
import os, multiprocessing
from data_feed import DataGenerator
from graph_model import Seq2SeqModel
from utils import create_local_model_path, create_local_log_path, retrieve_reverse_token_dict

def model_train():

    #pickle_file = 'full_dedup_scrambled_1_times_titles.pkl'
    pickle_file = 'full_dedup_scrambled_1_token_thres_8_titles.pkl'

    epoch_num = 4000
    batch_size = 512
    USE_RAW_RNN = True
    USE_GPU = True
    USE_BIDIRECTIONAL = True

    # PAD = 0 ## default padding is 0
    NUM_THREADS = 2 * multiprocessing.cpu_count() - 1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    #dataGen = DataGenerator(pickle_file_path, dual_outputs=USE_RAW_RNN)
    dataGen = DataGenerator(pickle_file_path, dual_outputs=True)
    batches = dataGen.generate_sequence(batch_size)

    model_config = {}
    model_config['restore_model'] = False
    model_config['eval_mode'] = False
    model_config['learning_rate'] = 0.001
    model_config['display_steps'] = 5000
    model_config['saving_steps'] = 5000
    model_config['embedding_size'] = 64
    model_config['hidden_units'] = 128

    #model_config['embedding_size'] = 512
    #model_config['hidden_units'] = 128


    #model_config['model_name'] = 'seq2seq_full_dedup_raw_rnn_scramble_1_token_thres_8'
    #model_config['model_name'] = 'seq2seq_full_dedup_raw_rnn_scramble_1_token_thres_8_embedding_128_hidden_256'
    #model_config['model_name'] = 'seq2seq_full_dedup_raw_rnn_scramble_1_token_thres_8_embedding_64_hidden_128_BiDirectional'
    model_config['model_name'] = 'seq2seq_model'

    model_config['batch_size'] = batch_size
    model_config['use_raw_rnn'] = USE_RAW_RNN
    model_config['BiDirectional'] = USE_BIDIRECTIONAL
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])

    if USE_GPU:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    reverse_token_dict = retrieve_reverse_token_dict(pickle_file_path)

    model = Seq2SeqModel(**model_config)
    num_batches = int(dataGen.data_size*epoch_num/model_config['batch_size'])
    print 'total #batches: {}, vocab_size: {}'.format(num_batches, model_config['vocab_size'])
    model.train(batches, num_batches, reverse_token_dict, dropout_input_keep_prob=0.5)


if __name__ == '__main__':
    model_train()

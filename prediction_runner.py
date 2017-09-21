import tensorflow as tf
import os, multiprocessing
from data_feed import DataGenerator
from graph_model import Seq2SeqModel
from utils import create_local_model_path, create_local_log_path

def model_predict():
    # pickle_file = 'processed_titles_data.pkl'
    pickle_file = 'lemmanized_no_stop_words_scrambled_titles.pkl'

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
    model_config['restore_model'] = True
    model_config['eval_mode'] = True
    model_config['model_name'] = 'seq2seq_dynamic_rnn_scrambled_lemmatized_content'
    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])

    '''
    model_config['batch_size'] = batch_size
    model_config['use_raw_rnn'] = USE_RAW_RNN
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['num_batches'] = int(dataGen.data_size * epoch_num / model_config['batch_size'])
    '''

    if USE_GPU:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    if USE_RAW_RNN:
        training_batch, target_batch = next(batches)
    else:
        target_batch = next(batches)

    model = Seq2SeqModel(**model_config)
    embedded_input_sets, encode_ouput_sets, hidden_state_sets = model.eval_by_batch(target_batch)
    print len(embedded_input_sets[0])
    print embedded_input_sets[0]
    print "\n"
    print len(embedded_input_sets[1])
    print embedded_input_sets[1]
    print "\n"
    print len(embedded_input_sets[2])
    print embedded_input_sets[2]

if __name__ == '__main__':
    model_predict()
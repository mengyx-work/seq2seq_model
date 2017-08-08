import os
import cPickle as pickle
import numpy as np
from data_preprocess import TOKEN_DICT, _GO, _EOS


def process_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


class DataGenerator(object):

    def __init__ (self, pickle_file_path):
        self._cur_index = 0
        with open(pickle_file_path, 'rb') as input_stream:
            self.data = pickle.load(input_stream)
        self.titles = self.data['titles']
        self.reverse_token_dict = self.data['reverse_token_dict']
        self.data_size = len(self.titles)
        self.vocab_size = max(self.reverse_token_dict.keys()) 
    
    def generate_sequence(self, batch_size):
        if batch_size >= 2 * self.data_size:
            raise ValueError("the batch_size can not be more than two times the data_size")
        
        while True:
            if self._cur_index + batch_size <= self.data_size:
                start_index = self._cur_index
                self._cur_index += batch_size
                yield self.titles[start_index : self._cur_index]
            else:
                start_index = self._cur_index
                self._cur_index = self._cur_index + batch_size - self.data_size
                batch_content = self.titles[start_index : self.data_size]
                batch_content.extend(self.titles[0 : self._cur_index])
                yield batch_content


def main():
    pickle_file = 'processed_titles.pkl'
    batch_size = 32
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path)
    batches = dataGen.generate_sequence(batch_size)

    batch = next(batches)
    #encoder_inputs_, encoder_inputs_length = process_batch(batch)
    encoder_inputs_, encoder_inputs_length = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
    decoder_targets_, decoder_targets_length = process_batch([sequence + [TOKEN_DICT[_EOS]] for sequence in batch])
    decoder_inputs_, decoder_inputs_length = process_batch([[TOKEN_DICT[_GO]] + sequence for sequence in batch])
    print decoder_inputs_
    print '\n \n'
    print encoder_inputs_

if __name__ == '__main__':
    main()
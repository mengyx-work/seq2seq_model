import cPickle as pickle
import numpy as np

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
                #yield self.titles[start_index : self.data_size] + self.titles[0 : self._cur_index]


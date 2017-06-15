import os
import numpy as np
import copy
import pickle

CODES = {'<PAD>':0, '<EOS>':1, '<UNK>':2, '<GO>': 3}



def load_file(path):
    files = os.path.join(path)
    with open(files, 'r', encoding='utf-8') as file_:
        data = file_.read()
    return data

def batch_data(inputs, target, batch_size):
    
    #batch data inputs and target along
    length = np.floor_divide(len(inputs), batch_size) # dividing the data by the size of the batch
    for i in range(length):
        start = i * batch_size
        input_path =  inputs[start:start + batch_size]
        target_path =  target[start:start + batch_size]
        yield np.array(pad_batch(input_path)), np.array(pad_batch(target_path))
        
##padding each sequence accordint to the longest sequence in the batch        
def pad_batch(batch):
    longest_seq = max([len(seq) for seq in batch])
    return [sequence + [CODES['<PAD>']] * (longest_seq - len(sequence)) for sequence in batch]
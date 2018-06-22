import numpy as np

def PadAndConcatChars(batch_test):
    batch_size = len(batch_test)
    shapes = np.empty(batch_size, dtype = int)
    for i in range(batch_size):
        shapes[i] = batch_test[i].shape[0]

    max_sentence_len = shapes.max()

    for i in range(batch_size):
        batch_test[i] = np.pad(batch_test[i], (0, max_sentence_len - batch_test[i].shape[0]), 'constant', constant_values=(0, 0))
    
    temp_chars = np.concatenate(batch_test).reshape(-1, max_sentence_len).transpose()
    
    return temp_chars

def PadAndConcatSpec(spectrograms):
    batch_size = len(spectrograms)
    specshapes = np.empty(batch_size, dtype = int)
    for i in range(batch_size):
        specshapes[i] = spectrograms[i].shape[1]

    max_spec_w = specshapes.max()
    
    for i in range(batch_size):
        spectrograms[i] = np.pad(spectrograms[i], ((0,0), (0, max_spec_w - spectrograms[i].shape[1])), 'constant', constant_values=(0, 0))

       
    return np.stack(spectrograms).transpose((2,0,1))
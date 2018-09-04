#coding=utf-8
import numpy as np

def random_embedding(vocab_size, embedding_dim):
    """
    随机初始化词向量
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def prepare_batch(data, labelg, labeli, memory, data_part, memory_part, batch_size):
    '''
    对一个epoch的数据和标签分batch并padding
    :param data: index_data
    :param label: sequence label
    :param memory: memory for each sample
    :return: padding_batch_data, padding_batch_label, batch_sequence_length
    '''
    num_batches_per_epoch = int((len(data) - 1) / batch_size)
    pad_data = []
    pad_data_part = []
    pad_labelg = []
    pad_labeli = []
    pad_memory = []
    pad_memory_part = []
    pad_seq_x = []
    pad_seq_m = []
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        in_data = data[start_index:end_index]
        in_data_part = data_part[start_index:end_index]
        in_labelg = labelg[start_index:end_index]
        in_labeli = labeli[start_index:end_index]
        in_memory = memory[start_index:end_index]
        in_memory_part = memory_part[start_index:end_index]
        max_len_x = max(len(x) for x in in_data)
        max_len_m = max(max(len(x) for x in a) for a in in_memory)
        bx = np.row_stack([np.pad(x, (0, max_len_x - len(x)), "constant", constant_values=0) for x in in_data])
        bx_part = np.row_stack([np.pad(x, (0, max_len_x - len(x)), "constant", constant_values=0) for x in in_data_part])
        by = np.vstack([np.pad(y, (0, max_len_x - len(y)), "constant", constant_values=0) for y in in_labelg])
        byi = np.vstack([np.pad(y, (0, max_len_x - len(y)), "constant", constant_values=0) for y in in_labeli])
        sq_len_x = np.array([len(x) for x in in_data])
        bm = np.array([np.row_stack([np.pad(x, (0, max_len_m - len(x)), "constant", constant_values=0) for x in a]) for a in in_memory])
        bm_part = np.array([np.row_stack([np.pad(x, (0, max_len_m - len(x)), "constant", constant_values=0) for x in a]) for a in in_memory_part])
        sq_len_m = np.array([np.array([len(x) for x in a]) for a in in_memory])
        pad_data.append(bx)
        pad_data_part.append(bx_part)
        pad_labelg.append(by)
        pad_labeli.append(byi)
        pad_memory.append(bm)
        pad_memory_part.append(bm_part)
        pad_seq_x.append(sq_len_x)
        pad_seq_m.append(sq_len_m)
    return pad_data, pad_data_part, pad_labelg, pad_labeli, pad_memory, pad_memory_part, pad_seq_x, pad_seq_m

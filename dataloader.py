import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import pandas as pd 
import numpy as np
import re
import os
import logging
import gzip
import _pickle as cPickle

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def clean_html(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext

def get_data(sen_set, label_set):
    """
        Create inputs, masks, labels in matrix form

    """
    input_ids = []
    for sens in tqdm(sen_set, desc='process split'):
        for sent in sens:
            encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
            input_ids.append(encoded_sent)
    
    inputs = pad_sequences(input_ids, maxlen=128, dtype="long", value=0, truncating="post", padding="post")        
    
    masks = []
    for sent in inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        masks.append(att_mask)   
    masks = np.array(masks)
    
    labels = np.concatenate(label_set)

    return inputs, masks, labels

# def processs_data(data_dir):
#     """ 
#         Process inputs, masks, labels from data directory
    
#     """

#     train_sen_set, train_label_set = [], []
#     test_sen_set, test_label_set = [], []
#     val_sen_set, val_label_set = [], []

#     for csv_f in tqdm(os.listdir(data_dir), desc='Read csv file'):
#         try:
#             df = pd.read_csv(os.path.join(data_dir, csv_f))
#         except Exception as e:
#             print(e)
#         else:
#             df = df.fillna(" ")
#             sentences = df.title.values + ' ' + df.description.values 
#             labels = df.storypoint.values
#             data_len = len(df)
#             n_train = int(0.6*data_len)
#             n_test = int(0.2*data_len)
#             train_sen_set.append(sentences[:n_train])
#             train_label_set.append(labels[:n_train])
#             val_sen_set.append(sentences[n_train:-n_test])
#             val_label_set.append(labels[n_train:-n_test])
#             test_sen_set.append(sentences[-n_test:])
#             test_label_set.append(labels[-n_test:])

#     train_inputs, train_masks, train_labels = get_data(train_sen_set, train_label_set)
#     val_inputs, val_masks, val_labels = get_data(val_sen_set, val_label_set)
#     test_inputs, test_masks, test_labels = get_data(test_sen_set, test_label_set)

#     return (train_inputs, train_masks, train_labels), \
#         (val_inputs, val_masks, val_labels), \
#         (test_inputs, test_masks, test_labels)

def processs_data(datafile, dictfile):
    """ 
        Process inputs, masks, labels from data directory
    
    """

    f_dict = gzip.open(dictfile, 'rb')
    dictionary = cPickle.load(f_dict,encoding='latin1')
    iv_dictionary = {v:k for k,v in dictionary.items()}
    iv_dictionary[0] = ''

    f_data = gzip.open(datafile, 'rb')
    train_t, train_d, train_labels, \
    valid_t, valid_d, val_labels, \
    test_t, test_d, test_labels = cPickle.load(f_data,encoding='latin1')

    train_sen_t_set = []
    for idx_set in train_t:
        train_sen_t_set.append(' '.join([iv_dictionary[i] for i in idx_set]))

    train_sen_d_set = []
    for idx_set in train_d:
        train_sen_d_set.append(' '.join([iv_dictionary[i] for i in idx_set]))

    val_sen_t_set = []
    for idx_set in valid_t:
        val_sen_t_set.append(' '.join([iv_dictionary[i] for i in idx_set]))
    
    val_sen_d_set = []
    for idx_set in valid_d:
        val_sen_d_set.append(' '.join([iv_dictionary[i] for i in idx_set]))

    test_sen_t_set = []
    for idx_set in test_t:
        test_sen_t_set.append(' '.join([iv_dictionary[i] for i in idx_set]))
    
    test_sen_d_set = []
    for idx_set in test_d:
        test_sen_d_set.append(' '.join([iv_dictionary[i] for i in idx_set]))

    train_t_inputs, train_t_masks, train_labels = get_data([train_sen_t_set], [train_labels])
    train_d_inputs, train_d_masks, train_labels = get_data([train_sen_d_set], [train_labels])

    val_t_inputs, val_t_masks, val_labels = get_data([val_sen_t_set], [val_labels])
    val_d_inputs, val_d_masks, val_labels = get_data([val_sen_d_set], [val_labels])

    test_t_inputs, test_t_masks, test_labels = get_data([test_sen_t_set], [test_labels])
    test_d_inputs, test_d_masks, test_labels = get_data([test_sen_d_set], [test_labels])

    return (train_t_inputs, train_t_masks, train_d_inputs, train_d_masks, train_labels), \
        (val_t_inputs, val_t_masks, val_d_inputs, val_d_masks, val_labels), \
        (test_t_inputs, test_t_masks,test_d_inputs, test_d_masks, test_labels)

def load_from_npz(processed_path):
    """ 
        Load inputs, masks, labels from numpy file
    
    """
    data = np.load(processed_path)
    return  data['inputs'], data['masks'], data['labels']

def get_split_dataloader(inputs, masks, inputs2, masks2, labels, batch_size):
    """ 
        Create dataloader for a triple (inputs, masks, labels)
    
    """
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    inputs2 = torch.tensor(inputs2)
    masks2 = torch.tensor(masks2)

    data = TensorDataset(inputs, masks, inputs2, masks2, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


# def get_all_dataloader(data_dir, processed_dir, datafile, dictfile, batch_size):
def get_all_dataloader(processed_dir, datafile, dictfile, batch_size):
    """ 
        Create 3 dataloaders 
    
    """
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
    processed_train_path = os.path.join(processed_dir, 'train.npz')
    processed_val_path = os.path.join(processed_dir, 'val.npz')
    processed_test_path = os.path.join(processed_dir, 'test.npz')

    # If already processed
    if os.path.isfile(processed_train_path) and os.path.isfile(processed_test_path) and os.path.isfile(processed_val_path):
        train_inputs, train_masks, train_labels = load_from_npz(processed_train_path)
        val_inputs, val_masks, val_labels = load_from_npz(processed_val_path)
        test_inputs, test_masks, test_labels = load_from_npz(processed_test_path)

    # Else process
    else:
        (train_t_inputs, train_t_masks, train_d_inputs, train_d_masks, train_labels), \
        (val_t_inputs, val_t_masks, val_d_inputs, val_d_masks, val_labels), \
        (test_t_inputs, test_t_masks,test_d_inputs, test_d_masks, test_labels) = processs_data(datafile, dictfile)
        
        # np.savez(processed_train_path, inputs=train_inputs, masks=train_masks, labels=train_labels)
        # np.savez(processed_val_path, inputs=val_inputs, masks=val_masks, labels=val_labels)
        # np.savez(processed_test_path, inputs=test_inputs, masks=test_masks, labels=test_labels)

    train_dataloader = get_split_dataloader(train_t_inputs, train_t_masks, train_d_inputs, train_d_masks, train_labels, batch_size)
    val_dataloader = get_split_dataloader(val_t_inputs, val_t_masks, val_d_inputs, val_d_masks, batch_size)
    test_dataloader = get_split_dataloader(test_t_inputs, test_t_masks,test_d_inputs, test_d_masks, test_labels, batch_size)

    return train_dataloader, val_dataloader, test_dataloader

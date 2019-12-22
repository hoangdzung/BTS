import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import pandas as pd 
import re
import os

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
    for sens in sen_set:
        for sent in sens:
            encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
            input_ids.append(encoded_sent[:511]+[encoded_sent[-1]])
    
    inputs = pad_sequences(input_ids, maxlen=512, dtype="long", value=0, truncating="post", padding="post")        
    
    masks = []
    for sent in inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        masks.append(att_mask)   
    masks = np.array(masks)
    
    labels = np.concatenate(label_set)

    return inputs, masks, labels

def processs_data(data_dir):
    """ 
        Process inputs, masks, labels from data directory
    
    """

    train_sen_set, train_label_set = [], []
    test_sen_set, test_label_set = [], []
    val_sen_set, val_label_set = [], []

    for csv_f in os.listdir(data_dir):
        try:
            df = pd.read_csv(os.path.join(data_dir, csv_f))
        except Exception as e:
            print(e)
        else:
            df = df.fillna(" ")
            sentences = df.title.values + ' ' + df.description.values 
            labels = df.storypoint.values
            data_len = len(df)
            n_train = int(0.6*data_len)
            n_test = int(0.2*data_len)
            train_sen_set.append(sentences[:n_train])
            train_label_set.append(labels[:n_train])
            val_sen_set.append(sentences[n_train:-n_test])
            val_label_set.append(labels[n_train:-n_test])
            test_sen_set.append(sentences[-n_test:])
            test_label_set.append(labels[-n_test:])

    train_inputs, train_masks, train_labels = get_data(train_sen_set, train_label_set)
    val_inputs, val_masks, val_labels = get_data(val_sen_set, val_label_set)
    test_inputs, test_masks, test_labels = get_data(test_sen_set, test_label_set)

    return (train_inputs, train_masks, train_labels), \
        (val_inputs, val_masks, val_labels), \
        (test_inputs, test_masks, test_labels)

def load_from_npz(processed_path):
    """ 
        Load inputs, masks, labels from numpy file
    
    """
    data = np.load(processed_path)
    return  data['inputs'], data['masks'], data['labels']

def get_split_dataloader(inputs, masks, labels, batch_size):
    """ 
        Create dataloader for a triple (inputs, masks, labels)
    
    """
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def get_all_dataloader(data_dir, processed_dir, batch_size):
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
        (train_inputs, train_masks, train_labels), (val_inputs, val_masks, val_labels), \
            (test_inputs, test_masks, test_labels) = processs_data(data_dir)
        
        np.savez(processed_train_path, inputs=train_inputs, masks=train_masks, labels=train_labels)
        np.savez(processed_val_path, inputs=val_inputs, masks=val_masks, labels=val_labels)
        np.savez(processed_test_path, inputs=test_inputs, masks=test_masks, labels=test_labels)

    train_dataloader = get_split_dataloader(train_inputs, train_masks, train_labels, batch_size)
    val_dataloader = get_split_dataloader(val_inputs, val_masks, val_labels, batch_size)
    test_dataloader = get_split_dataloader(test_inputs, test_masks, test_labels, batch_size)

    return train_dataloader, val_dataloader, test_dataloader

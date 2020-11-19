import pandas as pd

from config import cfg

import torch

def get_train(fold_idx):

    path = cfg.data_directory+'X_train_split_'+str(fold_idx)+'.csv'
    X = pd.read_csv(path)

    path = cfg.data_directory+'y_train_split_'+str(fold_idx)+'.csv'
    y = pd.read_csv(path)

    df = pd.concat([X,y],axis=1)

    return df

def get_dev(fold_idx):

    path = cfg.data_directory+'X_dev_split_'+str(fold_idx)+'.csv'
    X = pd.read_csv(path)

    path = cfg.data_directory+'y_dev_split_'+str(fold_idx)+'.csv'
    y = pd.read_csv(path)

    df = pd.concat([X,y],axis=1)

    return df

def merge_train_dev(train, dev):
    df = pd.concat([train,dev],axis=0).reset_index(drop=True)
    return df

def get_test():
    path = cfg.data_directory+'X_test.csv'
    df = pd.read_csv(path)

    return df

def get_Data(fold_idx):
    
    train = get_train(fold_idx)
    dev = get_dev(fold_idx)
    test = get_test()

    if cfg.final_train:
        train = merge_train_dev(train,dev)
        return train, train, test
    return train, dev, test

def calculate_max_len(sentences, tokenizer):

    max_len = 0

    for sent in sentences:

        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    
    return max_len

def get_tokenized(sentences,tokenizer,max_len):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sent in sentences:

        encoded_dict = tokenizer.encode_plus(
                                sent,       
                                add_special_tokens = True,
                                max_length = max_len,        
                                padding='max_length',
                                return_attention_mask = True,
                                return_token_type_ids = True,
                                return_tensors = 'pt', 
                                truncation = True
                )
    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)

    return input_ids, attention_masks, token_type_ids
    
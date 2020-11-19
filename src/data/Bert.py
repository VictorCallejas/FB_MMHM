import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from data.utils import get_Data, calculate_max_len, get_tokenized

from transformers import AutoTokenizer

from config import cfg, BertConfig

def getData(fold_idx):

    train, dev, test = get_Data(fold_idx)
    print('Data obtained')

    train, dev, test = getDataset(train,dev,test)

    sentences = []
    sentences.extend(train.full.values)
    if not cfg.final_train: 
        sentences.extend(dev.full.values)
    sentences.extend(test.full.values)

    tokenizer = AutoTokenizer.from_pretrained(BertConfig.model,do_lower=BertConfig.do_lower)

    if BertConfig.max_len_tokenized_sentence > -1:
        max_len = BertConfig.max_len_tokenized_sentence
        print('Using config max len: ', max_len)
    else:
        max_len = calculate_max_len(sentences,tokenizer)
        print('Using calculated  max len: ', max_len)

    train = getTensorDataset(train.id.values,train.full.values,train.label.values,tokenizer,max_len)
    dev = getTensorDataset(dev.id.values,dev.full.values,dev.label.values,tokenizer,max_len)
    test = getTensorDataset(test.id.values,test.full.values,None,tokenizer,max_len)

    print('Tensor Datasets created')

    train_dataloader, valid_dataloader, test_dataloader = getDataloader(train,dev,test)

    print('Dataloaders created')

    return train_dataloader, valid_dataloader, test_dataloader

def getDataset(train,dev,test):

    if (BertConfig.mode == 'image'):
        train['full'] = train[BertConfig.img_col]
        dev['full'] = dev[BertConfig.img_col]
        test['full'] = test[BertConfig.img_col]
    elif BertConfig.mode == 'image_caption':
        train['full'] = train[BertConfig.img_col]  + BertConfig.separator + train['text']
        dev['full'] = dev[BertConfig.img_col]  + BertConfig.separator + dev['text']
        test['full'] = test[BertConfig.img_col] + BertConfig.separator + test['text']
    else:
        train['full'] = train['text']
        dev['full'] = dev['text']
        test['full'] = test['text']

    train.full = train.full.astype(str)
    dev.full = dev.full.astype(str)
    test.full = test.full.astype(str)

    return train, dev, test

def getTensorDataset(ids, sentences, labels, tokenizer, max_len):

    input_ids , attention_masks, token_types_ids =  get_tokenized(sentences, tokenizer,max_len)

    if labels is None:
        return TensorDataset(torch.from_numpy(ids),input_ids,attention_masks,token_types_ids,torch.from_numpy(ids))

    return TensorDataset(torch.from_numpy(ids),input_ids,attention_masks,token_types_ids,torch.from_numpy(labels))

def getDataloader(train_dataset, valid_dataset,test_dataset):

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = BertConfig.train_batch_size
            )

    valid_dataloader = DataLoader(
                valid_dataset,  
                sampler = SequentialSampler(valid_dataset),
                batch_size = BertConfig.dev_batch_size
            )

    test_dataloader = DataLoader(
                test_dataset,  
                sampler = SequentialSampler(test_dataset),
                batch_size = BertConfig.dev_batch_size
            )
            
    return train_dataloader, valid_dataloader, test_dataloader
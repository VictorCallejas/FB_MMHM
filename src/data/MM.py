import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from data.utils import get_Data

from config import cfg, MMConfig

class VQAHatefulMemesDataset(Dataset):
    def __init__(self, ids, text, image, labels, frcnn_path, snip_img_path = True, test =False):
        self.ids = ids
        self.text = text
        self.image = image
        self.labels = np.zeros(len(self.ids)) if test else labels
        self.model = MMConfig.model
        self.frcnn_path = frcnn_path
        self.snip_img_path = snip_img_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        ids = self.ids[item]
        label = self.labels[item]

        if self.snip_img_path:
            img_path = self.image[item][4:-4]
        else:
            img_path = self.image[item]
        
        frcnn_outputs = np.load(self.frcnn_path+img_path+'.npy',allow_pickle=True).item()

        boxes = frcnn_outputs['boxes']
        features = frcnn_outputs['features']
        (img_w, img_h) = frcnn_outputs['size']

        text = self.text[item]

        if self.model == 'uniter-base' or self.model == 'uniter-large':
            boxes = self._uniterBoxes(boxes)
        if self.model == 'lxmert':
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h

        features = torch.tensor(features, dtype=torch.float)
        boxes = torch.tensor(boxes, dtype=torch.float)

        batch = {'ids': ids,
                'features': features,
                'boxes': boxes,
                'text': text,
                'label': torch.tensor(label, dtype=torch.float),
                }

        return batch 

    def _uniterBoxes(self, boxes):#uniter requires a 7-dimensiom beside the regular 4-d bbox
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1] #w
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0] #h
        new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5] #area
        return new_boxes 

def getData(fold_idx):

    train, dev, test = get_Data(fold_idx)
    print('Data obtained')

    train_dataset, dev_dataset, test_dataset = getDataset(train,dev,test)

    print('Tensor Datasets created')

    train_dataloader, valid_dataloader, test_dataloader = getDataloader(train_dataset,dev_dataset,test_dataset)

    print('Dataloaders created')

    return train_dataloader, valid_dataloader, test_dataloader

def getDataset(train,dev,test):

    if (MMConfig.mode == 'image'):
        train['full'] = train[MMConfig.img_col]
        dev['full'] = dev[MMConfig.img_col]
        test['full'] = test[MMConfig.img_col]
    elif MMConfig.mode == 'image_caption':
        train['full'] = train[MMConfig.img_col]  + MMConfig.separator + train['text']
        dev['full'] = dev[MMConfig.img_col]  + MMConfig.separator + dev['text']
        test['full'] = test[MMConfig.img_col] + MMConfig.separator + test['text']
    else:
        train['full'] = train['text']
        dev['full'] = dev['text']
        test['full'] = test['text']

    train.full = train.full.astype(str)
    dev.full = dev.full.astype(str)
    test.full = test.full.astype(str)

    train_dataset = VQAHatefulMemesDataset(train.id.values, train.full.values, train.img.values, train.label.values, MMConfig.frcnn_path)
    dev_dataset = VQAHatefulMemesDataset(dev.id.values, dev.full.values, dev.img.values, dev.label.values, MMConfig.frcnn_path)
    test_dataset = VQAHatefulMemesDataset(test.id.values, test.full.values, test.img.values, None, MMConfig.frcnn_path,test=True)

    return train_dataset, dev_dataset, test_dataset


def getDataloader(train_dataset, valid_dataset,test_dataset):

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = MMConfig.train_batch_size
            )

    valid_dataloader = DataLoader(
                valid_dataset,  
                sampler = SequentialSampler(valid_dataset),
                batch_size = MMConfig.dev_batch_size
            )

    test_dataloader = DataLoader(
                test_dataset,  
                sampler = SequentialSampler(test_dataset),
                batch_size = MMConfig.dev_batch_size
            )
            
    return train_dataloader, valid_dataloader, test_dataloader
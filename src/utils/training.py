import pandas as pd
import numpy as np

from utils.utils import save, get_device, make_submission

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn import metrics

from config import cfg

import matplotlib.pyplot as plt

import seaborn as sns

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import update_bn

from transformers.optimization import get_linear_schedule_with_warmup

model_config = None

def training(model, train_dataloader,valid_dataloader,test_dataloader, model_cfg,fold_idx=1):

    print("--------  ",str(fold_idx),"  --------")
    global model_config
    model_config = model_cfg

    device = get_device()
    model.to(device)
    
    if fold_idx==1 : print('CONFIG: ')
    if fold_idx==1 : print([(v,getattr(model_config,v)) for v in dir(model_config) if v[:2] != "__"])
    if fold_idx==1 : print('MODEL: ',model)

    epochs = model_config.epochs

    if model_config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = float(model_config.lr),
            eps = float(model_config.eps),
            weight_decay = float(model_config.weight_decay)
        )
    elif model_config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr = float(model_config.lr)
        )

    if model_config.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                        num_warmup_steps = int(model_config.warmup_steps),
                        num_training_steps = len(train_dataloader)*epochs)
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()

    swa_model = AveragedModel(model)
    if model_config.swa_scheduler == 'linear':
        swa_scheduler = SWALR(optimizer, swa_lr=float(model_config.lr))
    else:
        swa_scheduler = CosineAnnealingLR(optimizer, T_max=100)

    print('TRAINING...')

    training_stats = []

    best_dev_auc = float('-inf')

    with tqdm(total=epochs,leave=False) as pbar:
        for epoch_i in range(0, epochs):

            if epoch_i >= int(model_config.swa_start):
                update_bn(train_dataloader,swa_model)
                train_auc, train_acc, avg_train_loss = train(model,train_dataloader,device, criterion,optimizer)
                swa_model.update_parameters(model)
                swa_scheduler.step()
                update_bn(valid_dataloader,swa_model)
                valid_auc, valid_acc, avg_dev_loss, dev_d = valid(swa_model,valid_dataloader,device,criterion)
            else:
                train_auc, train_acc, avg_train_loss = train(model,train_dataloader,device, criterion,optimizer,scheduler=scheduler)
                valid_auc, valid_acc, avg_dev_loss, dev_d = valid(model,valid_dataloader,device,criterion)
            if cfg.final_train:
                valid_auc = 0
                valid_acc = 0
                avg_dev_loss = 0

            add_stats(training_stats,avg_train_loss,avg_dev_loss,train_acc,train_auc,valid_acc,valid_auc)
            
            if (cfg.final_train & (epoch_i == epochs-1)) | (not cfg.final_train & (valid_auc > best_dev_auc)):
                best_dev_auc = valid_auc
                if epoch_i >= int(model_config.swa_start):
                    update_bn(test_dataloader,swa_model)
                    test_d = gen_test(swa_model,test_dataloader,device)
                    save(fold_idx,swa_model,optimizer,dev_d,test_d,valid_auc)
                else:
                    test_d = gen_test(model,test_dataloader,device)
                    save(fold_idx,model,optimizer,dev_d,test_d,valid_auc)
                
            pbar.update(1)
    
    print('TRAINING COMPLETED')

    # Show training results
    col_names = ['train_loss','train_acc','train_auc','dev_loss', 'dev_acc','dev_auc']
    training_stats = pd.DataFrame(training_stats,columns=col_names)
    print(training_stats.head(epochs))
    plot_training_results(training_stats,fold_idx)

    # If config, get best model and make submission
    if cfg.run['submission'] == True:
        make_submission(model,test_dataloader)


def train(model, train_dataloader,device,criterion,optimizer,scheduler=None):

    total_train_loss = 0
    model.train()

    logits = []
    ground_truth = []
    loss = 0

    optimizer.zero_grad()
    for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

        with torch.cuda.amp.autocast(enabled=False):
            _,b_logits,_ ,b_labels = model(batch,device)

        logits.extend(b_logits)
        ground_truth.extend(b_labels)
                
        loss = criterion(b_logits.float().squeeze(),b_labels.float()) / int(model_config.gradient_accumulation_steps)
        loss.backward()

        total_train_loss += loss.item()

        if (step+1) % int(model_config.gradient_accumulation_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_config.clipping_grad_norm))
            optimizer.step()
            if scheduler is not None: scheduler.step()
            optimizer.zero_grad()
            loss = 0

    y_probas = nn.Sigmoid()(torch.Tensor(logits))
    y_labels = y_probas.round()

    train_auc = metrics.roc_auc_score(ground_truth,y_probas)
    train_acc = metrics.accuracy_score(ground_truth,y_labels)
    avg_train_loss = total_train_loss/len(train_dataloader)

    return train_auc, train_acc, avg_train_loss

def valid(model,valid_dataloader,device,criterion):
            
            model.eval()

            total_dev_loss = 0
            loss = 0

            logits = []
            ground_truth = []
            ids = []
            features = []
            
            for step, batch in tqdm(enumerate(valid_dataloader),total=len(valid_dataloader)):
                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad(): 
                        b_ids, b_logits, feats, b_labels = model(batch,device)

                ids.extend(b_ids.detach().numpy())
                features.extend(feats.detach().numpy())

                logits.extend(b_logits)
                ground_truth.extend(b_labels)

                loss = criterion(b_logits.squeeze().float(),b_labels.float()) / int(model_config.gradient_accumulation_steps)
                total_dev_loss += loss.item()

                if (step+1) % int(model_config.gradient_accumulation_steps) == 0:
                    loss = 0

            y_probas = nn.Sigmoid()(torch.Tensor(logits))
            y_labels = y_probas.round()

            valid_auc = metrics.roc_auc_score(ground_truth,y_probas)
            valid_acc = metrics.accuracy_score(ground_truth,y_labels)
            avg_dev_loss = total_dev_loss/len(valid_dataloader)

            dev_d = {
                'id':ids,
                'probas':y_probas,
                'pred_labels':y_labels,
                'features':features
            }

            return valid_auc, valid_acc, avg_dev_loss, dev_d

def gen_test(model,test_dataloader,device):

            model.eval()

            logits = []
            ids = []
            features = []
            
            for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad(): 
                        b_ids, b_logits, feats, _ = model(batch,device)

                ids.extend(b_ids.detach().numpy())
                features.extend(feats.detach().numpy())

                logits.extend(b_logits)

            y_probas = nn.Sigmoid()(torch.Tensor(logits))
            y_labels = y_probas.round()

            test_d = {
                'id':ids,
                'probas':y_probas,
                'pred_labels':y_labels,
                'features':features
            }

            return test_d
 
def add_stats(training_stats,avg_train_loss,avg_dev_loss,train_acc,train_auc,test_acc,test_auc):

    training_stats.append(
        {
            'train_loss': avg_train_loss,
            'dev_loss': avg_dev_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'dev_acc': test_acc,
            'dev_auc': test_auc
        }
    )
    print('train_loss ', avg_train_loss,
        ' dev_loss ', avg_dev_loss,
        ' train_acc ', train_acc,
        ' train_auc ', train_auc,
        ' dev_acc ', test_acc,
        ' dev_auc ', test_auc
    )

def plot_training_results(results,fold_idx):
        make_plot(results['train_loss'],results['dev_loss'],fold_idx,x='Loss')
        make_plot(results['train_acc'],results['dev_acc'],fold_idx,x='Accuracy')
        make_plot(results['train_auc'],results['dev_auc'],fold_idx,x='Area Under Curve')


def make_plot(train_serie, dev_serie,fold_idx,x,y='Epoch',l_s1='Training',l_s2='Validation',title="Training & Validation"):

    from utils.utils import RUN_FOLDER # Must be dynamic, if not inited, equal to None

    sns.set(style='darkgrid')

    sns.set(font_scale=1.5)

    plt.close()

    plt.rcParams["figure.figsize"] = (12,6)

    plt.plot(train_serie, 'b-o', label=l_s1)
    plt.plot(dev_serie, 'g-o', label=l_s2)

    plt.title(title)
    plt.xlabel(y)
    plt.ylabel(x)
    plt.legend()
    plt.xticks(np.arange(train_serie.shape[0]))

    #plt.show() # Causes interruption
    plt.savefig(RUN_FOLDER+'/'+str(fold_idx)+x)
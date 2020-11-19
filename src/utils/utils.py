import pandas as pd
import numpy as np

import torch

import os
from datetime import datetime
import sys

from config import cfg

from data.utils import get_test

DIR_PATH = '../artifacts/'
RUN_TYPE = None
FOLDER_NAME = None
RUN_FOLDER = None

def set_deterministic():
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

class Logger():
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open(path,mode='x')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        pass 

def config_run_logs():

    global RUN_TYPE

    RUN_TYPE = cfg.run['setup']

    #now = datetime.now()

    global FOLDER_NAME
    FOLDER_NAME = cfg.run['model_name']#now.strftime("%d_%m_%Y_%H_%M")

    global RUN_FOLDER

    RUN_FOLDER = DIR_PATH+RUN_TYPE+'/'+FOLDER_NAME

    try:
        os.makedirs(RUN_FOLDER)
    except:
        error = True
        print('Folder already exists')
        i = 1
        while error:
            RUN_FOLDER = RUN_FOLDER + str(i)
            try:
                os.makedirs(RUN_FOLDER)
                error = False
            except:
                i+=1
                RUN_FOLDER = RUN_FOLDER[:-1]

    print('Run folder created: ', RUN_FOLDER)
    redirect_output()

def redirect_output():
    try:
        sys.stdout = Logger(path=RUN_FOLDER+'/out.log')
    except:
        print('Log File exists')
    print('BASH OUTPUT REDIRECTED TO LOG FILE')


def save(f_idx,model,optimizer,dev,test,dev_auc):

    if not os.path.isdir(RUN_FOLDER+'/model'):
        os.makedirs(RUN_FOLDER+'/model')
    if not os.path.isdir(RUN_FOLDER+'/preds'):
        os.makedirs(RUN_FOLDER+'/preds')

    path = RUN_FOLDER+'/model/'+str(f_idx)+'.ckpt'

    torch.save({
        #'text_model_name':model.text_model_name,
        'model':model.state_dict(),
        'opt':optimizer.state_dict(),
        'dev_auc':dev_auc
    },path)

    path = RUN_FOLDER+'/preds/'
    np.savez(str(path+'dev_'+str(f_idx)+'.npz'),**dev,allow_pickle=True)
    np.savez(str(path+'test_'+str(f_idx)+'.npz'),**test,allow_pickle=True)

    print('Model and preds saved on ',RUN_FOLDER)

def make_submission(model, test_dataloader):

    print('MAKING SUBMISSION')

    model.eval()

    logits = []

    device = get_device()

    for batch in test_dataloader:

        with torch.no_grad():
            _ , b_logits, _ , _ = model(batch,device)
            logits.extend(b_logits)

    y_probas = torch.nn.Sigmoid()(torch.Tensor(logits))
    y_labels = y_probas.round()

    col_names = ['id','proba','label']

    subm = pd.DataFrame(columns=col_names)
    subm.id = get_test().id
    subm.label = y_labels.detach().numpy().astype(np.int64)
    subm.proba = y_probas.detach().numpy().astype(np.float64)

    print(subm.shape)
    print(subm.head())

    file_path = RUN_FOLDER+'/submission.csv'
    subm.to_csv(file_path,index = False, header=True,sep=',',encoding='utf-8-sig')

    print('SUBMISSION  DONE.')

def get_device():

    if torch.cuda.is_available() & (cfg.run['device'] == 'cuda'):    
    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        print('Properties:', torch.cuda.get_device_properties(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device
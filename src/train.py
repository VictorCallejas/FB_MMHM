import warnings
warnings.filterwarnings('ignore')

import  models.Bert, models.MM
import data.Bert, data.MM
import config.BertConfig, config.MMConfig

from utils.utils import config_run_logs, set_deterministic

from utils.training import training
from config import cfg

from tqdm import tqdm


def main():

    #INIT RUN
    config_run_logs()
    set_deterministic()

    print('RUN ',cfg.run['setup'].upper())

    if cfg.run['model'] == 'Bert':
        model_config = config.BertConfig
        for fold_idx in tqdm(range(1,int(1+cfg.run['n_folds'])),total=cfg.run['n_folds']):
            model = models.Bert.BertModel()
            train_dataloader, valid_dataloader, test_dataloader = data.Bert.getData(fold_idx)
            training(model,train_dataloader,valid_dataloader,test_dataloader,model_config,fold_idx=fold_idx)
    
    if cfg.run['model'] == 'MM':
        model_config = config.MMConfig
        for fold_idx in tqdm(range(1,int(1+cfg.run['n_folds'])),total=cfg.run['n_folds']):
            model = models.MM.VQA_Model()
            train_dataloader, valid_dataloader, test_dataloader = data.MM.getData(fold_idx)
            training(model,train_dataloader,valid_dataloader,test_dataloader,model_config,fold_idx=fold_idx)

    print('--- PROGRAM END ---')


if __name__ == "__main__":
    main() 
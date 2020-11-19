import torch
import torch.nn as nn

from config import MMConfig

from utils.mm.params import *
args.max_seq_length = MMConfig.max_seq_length
args.model = MMConfig.model
args.pretrained_weights = MMConfig.pretrained_weights

from utils.mm.entry import *
from utils.mm.file_utils import *
from utils.mm.modeling import *
from utils.mm.optimization import *
from utils.mm.tokenization import *
from utils.mm.vqa_model import *


class VQA_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.text_model_name = MMConfig.model

        self.vqa_model = VQAModel(num_answers = MMConfig.hidden_size, args=args,model=self.text_model_name)
        self.vqa_model.encoder.load(args.pretrained_weights)
        
        self.cls = nn.Linear(MMConfig.hidden_size, 1)

    def forward(self, batch, device):

        ids = batch['ids']
        labels = batch['label']

        features = batch['features'].to(device)
        boxes = batch['boxes'].to(device)
        texts = batch['text']
        
        feats = self.vqa_model(features,boxes,texts)
     
        x = self.cls(feats)

        return ids,x.cpu(),feats.cpu(), labels
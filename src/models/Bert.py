import torch
import torch.nn as nn

from transformers import AutoModel

from config import BertConfig

class BertModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.text_model_name = BertConfig.model
        self.text_model = AutoModel.from_pretrained(BertConfig.model)
        self.token_type_ids = BertConfig.token_type_ids
        """
        for p in self.text_model.embeddings.parameters():
            p.requires_grad = False   
        """
        self.cls = nn.Linear(BertConfig.hidden_size, 1)

    def forward(self, batch, device):

        ids = batch[0]
        labels = batch[4]

        input_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        if self.token_type_ids:
            token_type_ids = batch[3].to(device)
            feats = self.text_model(input_ids,attention_mask,token_type_ids)[0][:,0,:]
        else:
            feats = self.text_model(input_ids,attention_mask)[0][:,0,:]

        x = self.cls(feats)

        return ids,x.cpu(),feats.cpu(), labels
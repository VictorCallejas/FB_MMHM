# coding=utf-8
# Copyleft 2019 project LXRT.
# copied from LXRT with modifications
import torch.nn as nn

from utils.mm.entry import LXRTEncoder, UniterEncoderBase, UniterEncoderLarge, VBEncoderBase, VBEncoderLarge
from utils.mm.modeling import BertLayerNorm, GeLU


class VQAModel(nn.Module):
    def __init__(self, num_answers, args,model = 'uniter-base'):
        super().__init__()
        self.model = model
        # Build LXRT encoder
        if model == 'lxmert':
            self.encoder = LXRTEncoder(args)  
        elif model == 'visualbert-base':
            self.encoder = VBEncoderBase(args)
        elif model == 'visualbert-large':
            self.encoder = VBEncoderLarge(args)
        elif model == 'uniter-base':
            self.encoder = UniterEncoderBase(args)
        elif model == 'uniter-large':
            self.encoder = UniterEncoderLarge(args)
        
        hid_dim = self.encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        if self.model == 'lxmert':
            x = self.encoder(sent, (feat, pos))
        elif self.model == 'visualbert-base':
            x = self.encoder(sent, feat)
        elif self.model == 'visualbert-large':
            x = self.encoder(sent, feat)
        elif self.model == 'uniter-base':
            x = self.encoder(sent, feat, pos)
        elif self.model == 'uniter-large':
            x = self.encoder(sent, feat, pos)
        logit = self.logit_fc(x)

        return logit
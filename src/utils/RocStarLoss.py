# Modificated version of https://github.com/iridiumblue/roc-star
# We do not use it on our final models, this gives worse results

import torch
from torch.nn.modules.loss import _Loss

class RocStarLoss(_Loss):
    """Smooth approximation for ROC AUC
    """
    def __init__(self,device, delta = 1.0, sample_size = 1000, sample_size_gamma = 1000, update_gamma_each=50):
        r"""
        Args:
            delta: Param from article
            sample_size (int): Number of examples to take for ROC AUC approximation
            sample_size_gamma (int): Number of examples to take for Gamma parameter approximation
            update_gamma_each (int): Number of steps after which to recompute gamma value.
        """
        super().__init__()
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_gamma = sample_size_gamma
        self.update_gamma_each = update_gamma_each
        self.steps = 0
        size = max(sample_size, sample_size_gamma)

        # Randomly init labels
        self.y_pred_history = torch.rand((size, 1))
        self.y_true_history = torch.randint(2, (size, 1))

        self.device = device
        

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Tensor of model predictions in [0, 1] range. Shape (B x 1)
            y_true: Tensor of true labels in {0, 1}. Shape (B x 1)
        """
        #y_pred = _y_pred.clone().detach()
        #y_true = _y_true.clone().detach()
        if self.steps % self.update_gamma_each == 0:
            self.update_gamma()
        self.steps += 1
        
        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]
        
        # Take last `sample_size` elements from history
        y_pred_history = self.y_pred_history[- self.sample_size:]
        y_true_history = self.y_true_history[- self.sample_size:]
        
        positive_history = y_pred_history[y_true_history > 0].to(self.device)
        negative_history = y_pred_history[y_true_history < 1].to(self.device)

        self.gamma = self.gamma.to(self.device)
        
        if positive.size(0) > 0:
            diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
            loss_positive = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_positive = 0
 
        if negative.size(0) > 0:
            diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
            loss_negative = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_negative = 0
            
        loss = loss_negative + loss_positive
        
        # Update FIFO queue
        batch_size = y_pred.size(0)

        self.y_pred_history = self.y_pred_history.to(self.device)
        self.y_true_history = self.y_true_history.to(self.device)

        self.y_pred_history = torch.cat((self.y_pred_history[batch_size:], y_pred.clone().detach()))
        self.y_true_history = torch.cat((self.y_true_history[batch_size:], y_true.clone().detach()))
        return loss

    def update_gamma(self):
        # Take last `sample_size_gamma` elements from history
        y_pred = self.y_pred_history[- self.sample_size_gamma:]
        y_true = self.y_true_history[- self.sample_size_gamma:]
        
        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]
        
        # Create matrix of size sample_size_gamma x sample_size_gamma
        diff = positive.view(-1, 1) - negative.view(1, -1)
        AUC = (diff > 0).type(torch.float).mean()
        num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)
        
        # Adjuct gamma, so that among correct ordered samples `delta * num_wrong_ordered` were considered
        # ordered incorrectly with gamma added
        correct_ordered = diff[diff > 0].flatten().sort().values
        idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered)-1)
        self.gamma = correct_ordered[idx]
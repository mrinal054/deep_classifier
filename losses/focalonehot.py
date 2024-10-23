import torch

class FocalLossOneHot(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLossOneHot, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to inputs
        inputs_soft = torch.softmax(inputs, dim=1)
        
        # Compute log softmax
        logpt = torch.log(inputs_soft)
        
        # Multiply by one-hot encoded targets
        loss = -targets * logpt
        
        # Get probabilities
        pt = inputs_soft * targets
        
        # Focal Loss term
        F_loss = self.alpha * (1 - pt) ** self.gamma * loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

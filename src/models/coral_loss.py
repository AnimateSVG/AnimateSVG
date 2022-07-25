import torch.nn.functional as F
import torch


def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    """ Compute the CORAL loss described in Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008.

    Args:
        logits (torch.Tensor): Outputs of the CORAL layer (shape: (num_examples, num_classes-1)).
        levels (torch.Tensor): True labels represented as extended binary vectors (shape: (num_examples, num_classes-1)).
        importance_weights (torch.Tensor): Optional weights for the different labels in levels (shape: (num_classes-1,)).
        reduction (str): If 'mean' or 'sum', returns the averaged or summed loss value across all data points (rows) in
                    logits. If None, returns a vector of shape (num_examples,).

    Returns:
        torch.Tensor: A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
                      or a loss value for each data record (if `reduction=None`).

    """
    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as logits (%s). "
                         % (logits.shape, levels.shape))

    term1 = (F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss
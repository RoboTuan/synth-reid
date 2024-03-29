from __future__ import division, print_function, absolute_import
import torch
import torch.nn as nn


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.

    Examples::
        >>> from torchreid import metrics
        >>> metrics.accuracy(output, target)
    """
    maxk = max(topk)
    batch_size = target.size(0)
    sm = nn.Softmax(dim=1)

    if isinstance(output, (tuple, list)):
        if len(output) == 2:  # triplet loss
            output = output[0]
        else:  # pcb
            scores = torch.zeros(output[0].shape)
            scores = scores.cuda()
            for part_output in output:
                scores += sm(part_output)
            output = scores

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

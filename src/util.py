import torch


def perm_compose(p, q):
    return torch.LongTensor([p[q[i]] for i in range(len(p))])


def perm_invert(p):
    pinv = torch.zeros(len(p)).long()
    for i in range(len(p)):
        pinv[p[i]] = i
    return pinv

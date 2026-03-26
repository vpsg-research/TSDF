import copy
import numpy as np
from collections.abc import Iterable
from scipy.stats import truncnorm
import torch.nn.functional as Function
import sklearn.metrics as skm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
import os
import time

class Wasserstein_loss(nn.Module):
    def __init__(self) -> None:
        super(Wasserstein_loss, self).__init__()
    def forward(self, input, target):
        return torch.mean(input*target)

try:
    import defenses.smoothing as smoothing
except:
    import attgan.defenses.smoothing as smoothing


class get_all_features(object):
    def __init__(self, model=None, device='cuda', epsilon=0.05, args=None):
        """
        epsilon: magnitude of attack
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.MSELoss().to(device)
        self.wasserstein_loss = Wasserstein_loss().to(device)
        self.device = device

        self.up = torch.zeros([1, 3, 256, 256]).to(self.device)

    def get_attgan_features(self, X_input, X_att, attgan, rand_param):
        # attribute augmentation, default=No
        q = rand_param
        attr_att = torch.tensor(np.random.uniform(-0.5, 0.5, X_att.size()).astype('float32')).to(self.device)
        new_attr = (X_att * 0.8 + attr_att * 0.2) if q > 0.2 else X_att

        output, middle = attgan.G(X_input, new_attr)
        attgan.G.zero_grad()
        return output, middle

    def get_stargan_features(self, X_input, c_trg, model, rand_param):
        output, feats, middle = model.forward_my_attack(X_input, c_trg, rand_param)
        model.zero_grad()
        return output, middle

    def get_atggan_features(self, X_input, c_trg, model, rand_param):
        output, _, _, middle = model.forward_my_attack(X_input, c_trg, rand_param)
        model.zero_grad()
        return output, middle

    def get_hisd_features(self, X_input, reference, F, T, G, E, gen, rand_param):
        c = E(X_input)  # Feature extractor
        c_trg = c
        s_trg = F(reference, 1)  # reference

        # attribute augmentation, default=No
        size = s_trg.size()
        q = rand_param
        s_trg = (s_trg * 0.8 + torch.tensor(np.random.uniform(s_trg.clone().detach().cpu().numpy().min(),
                                                              s_trg.clone().detach().cpu().numpy().max(), size).astype('float32')).to(self.device) * 0.2) if q > 0.2 else s_trg

        c_trg = T(c_trg, s_trg, 1)
        x_trg = G(c_trg)
        gen.zero_grad()

        return x_trg, c

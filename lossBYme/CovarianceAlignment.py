import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


ce_weight = None
criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean', ignore_index=-1).cuda()
criterion_CA = nn.MSELoss()

def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B


def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape

    B, C, H, W = f_map1.shape
    HW = H * W

    if eye is None:
        eye = torch.eye(C).cuda()

    # feature map shape : (B,C,H,W) -> (B,C,HW)
    f_map1 = f_map1.contiguous().view(B, C, -1)
    f_map2 = f_map2.contiguous().view(B, C, -1)

    # f_cor shape : (B, C, C)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW - 1) + (eps * eye)

    return f_cor, B

def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape

    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).cuda()

    # get diagonal values of covariance matrix
    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).cuda()
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    diag_loss = diag_loss / B

    return diag_loss

def CML_CCL(k_arr, q_arr):
    CML = torch.FloatTensor([0]).cuda()
    CCL = torch.FloatTensor([0]).cuda()

    for N, f_maps in enumerate(zip(k_arr, q_arr)):
        k_maps, q_maps = f_maps
        # detach original images
        k_maps = k_maps.detach()
        k_cor, _ = get_covariance_matrix(k_maps)
        q_cor, _ = get_covariance_matrix(q_maps)
        cov_loss = criterion_CA(k_cor, q_cor)
        crosscov_loss = cross_whitening_loss(k_maps, q_maps)
        CML = CML + cov_loss
        CCL = CCL + crosscov_loss
    CML = CML / len(k_arr)
    CCL = CCL / len(k_arr)

    return CML, CCL

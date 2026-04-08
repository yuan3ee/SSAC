import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time

class Class_PixelNCELoss(nn.Module):
    def __init__(self, numclass):
        super(Class_PixelNCELoss, self).__init__()

        self.ignore_label = 255

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.max_classes = numclass
        self.max_views = 50

    # reshape label or prediction
    def resize_label(self, labels, HW):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 HW, mode='nearest')
        labels = labels.squeeze(1).long()

        return labels

    def _hard_anchor_sampling(self, X_q, X_k, y_hat, y):
        # X : Feature map, shape:(B, h*w, C), y_hat : label, shape:(B, h*w), y : prediction, shape:(B, H*W?)
        batch_size, feat_dim = X_q.shape[0], X_q.shape[-1]

        classes = []
        num_classes = []
        total_classes = 0
        # 한 배치 내의 이미지들에 대한 label들로부터 존재하는 class들 골라내기
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)  # 텐서에서 중복된 요소 제거하여 존재하는 고유요소들 반환
            this_classes = [x for x in this_classes if x != self.ignore_label]  # ignore label 제거
            this_classes = [x for x in this_classes if
                            (this_y == x).nonzero().shape[0] > self.max_views]  # class가 일정 개수 이상인 경우만 골라내기

            classes.append(this_classes)

            total_classes += len(this_classes)
            num_classes.append(len(this_classes))

        # return none if there is no class in the image
        if total_classes == 0:
            return None, None, None

        n_view = self.max_views

        # output tensors
        X_q_ = torch.zeros((batch_size, self.max_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_k_ = torch.zeros((batch_size, self.max_classes, n_view, feat_dim), dtype=torch.float).cuda()

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            this_indices = []

            # if there is no class in the image, randomly sample patcthes
            if len(this_classes) == 0:
                indices = torch.arange(X_q.shape[1], device=X_q.device)
                perm = torch.randperm(X_q.shape[1], device=X_q.device)
                indices = indices[perm[:n_view * self.max_classes]]
                indices = indices.view(self.max_classes, -1)

                X_q_[ii, :, :, :] = X_q[ii, indices, :]
                X_k_[ii, :, :, :] = X_k[ii, indices, :]

                continue

            # referecne : https://github.com/tfzhou/ContrastiveSeg/tree/main
            for n, cls_id in enumerate(this_classes):

                if n == self.max_classes:
                    break

                # sample hard pathces(wrong prediction) and easy pathces(correct prediction)
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_q_[ii, n, :, :] = X_q[ii, indices, :].squeeze(1)
                X_k_[ii, n, :, :] = X_k[ii, indices, :].squeeze(1)

                this_indices.append(indices)

                # fill the spare space with random pathces
            if len(this_classes) < self.max_classes:
                this_indices = torch.stack(this_indices)
                this_indices = this_indices.flatten(0, 1)

                num_remain = self.max_classes - len(this_classes)
                all_indices = torch.arange(X_q.shape[1], device=X_q[0].device)
                left_indices = torch.zeros(X_q.shape[1], device=X_q[0].device, dtype=torch.uint8)
                left_indices[this_indices] = 1
                left_indices = all_indices[~left_indices]

                perm = torch.randperm(len(left_indices), device=X_q[0].device)

                indices = left_indices[perm[:n_view * num_remain]]
                indices = indices.view(num_remain, -1)

                X_q_[ii, n + 1:, :, :] = X_q[ii, indices, :]
                X_k_[ii, n + 1:, :, :] = X_k[ii, indices, :]

        return X_q_, X_k_, num_classes

    def _contrastive(self, feats_q_, feats_k_):
        # feats shape : (B, nc, N, C)
        batch_size, num_classes, n_view, patch_dim = feats_q_.shape
        num_patches = batch_size * num_classes * n_view

        # feats shape : (B*nc*N, 1, C)
        feats_q_ = feats_q_.contiguous().view(num_patches, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(num_patches, -1, patch_dim)

        # logit_positive : same positive patches between key and query
        # shape : (B * nc * N , 1)
        l_pos = torch.bmm(
            feats_q_, feats_k_.transpose(2, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # feats shape : (B, nc*N, C)
        feats_q_ = feats_q_.contiguous().view(batch_size, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(batch_size, -1, patch_dim)
        n_patches = feats_q_.shape[1]

        # logit negative shape : (B, nc*N, nc*N)
        l_neg_curbatch = torch.bmm(feats_q_, feats_k_.transpose(2, 1))

        # exclude same class patches
        diag_block = torch.zeros((batch_size, n_patches, n_patches), device=feats_q_.device, dtype=torch.uint8)
        for i in range(num_classes):
            diag_block[:, i * n_view:(i + 1) * n_view, i * n_view:(i + 1) * n_view] = 1

        l_neg_curbatch = l_neg_curbatch[~diag_block].view(batch_size, n_patches, -1)

        # logit negative shape : (B*nc*N, nc*(N-1))
        l_neg = l_neg_curbatch.view(num_patches, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feats_q_.device))

        return loss

    def forward(self, feats_q, feats_k, labels=None, predict=None):
        B, C, H, W = feats_q.shape

        # resize label and prediction
        labels = self.resize_label(labels, (H, W))
        predict = self.resize_label(predict, (H, W))

        labels = labels.contiguous().view(B, -1)
        predict = predict.contiguous().view(B, -1)

        # change axis
        feats_q = feats_q.permute(0, 2, 3, 1)
        feats_q = feats_q.contiguous().view(feats_q.shape[0], -1, feats_q.shape[-1])

        feats_k = feats_k.detach()

        feats_k = feats_k.permute(0, 2, 3, 1)
        feats_k = feats_k.contiguous().view(feats_k.shape[0], -1, feats_k.shape[-1])

        # sample patches
        feats_q_, feats_k_, num_classes = self._hard_anchor_sampling(feats_q, feats_k, labels, predict)

        if feats_q_ is None:
            loss = torch.FloatTensor([0]).cuda()
            return loss

        loss = self._contrastive(feats_q_, feats_k_)

        del labels

        return loss


class Disentangle_Contrast(nn.Module):
    def __init__(self, numclass):
        super(Disentangle_Contrast, self).__init__()

        self.temperature = 0.1
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.num_patch = 20
        self.num_classes = numclass
        self.max_samples = 1000

    # reshape label or prediction
    def reshape_map(self, map, shape):

        map = map.unsqueeze(1).float().clone()
        map = torch.nn.functional.interpolate(map, shape, mode='nearest')
        map = map.squeeze(1).long()

        return map

    def _contrastive(self, pos_q, pos_k, neg):
        num_patch, _, patch_dim = pos_q.shape

        # l_pos shape : (num_patch, 1)
        l_pos = torch.bmm(pos_q, pos_k.transpose(2, 1))
        l_pos = l_pos.view(num_patch, 1)

        # l_neg shape : (num_patch, negative_size)
        l_neg = torch.bmm(pos_q, neg.transpose(2, 1))
        l_neg = l_neg.view(num_patch, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=pos_q.device))

        return loss

    def Disentangle_Sampler(self, correct_maps, feats_q, feats_k, predicts, predicts_j, labels):
        B, HW, C = feats_q.shape
        start_ts = time.time()

        # X_pos_q : anchors, X_pos_k : positives, X_neg : negatives
        X_pos_q = []
        X_pos_k = []
        X_neg = []

        for ii in range(B):
            img_sample_ts = time.time()
            M = correct_maps[ii]
            # indices : wrong prediction location in query features
            indices = (M == 1).nonzero()

            classes_labels = torch.unique(labels[ii])
            classes_wrong = torch.unique(predicts_j[ii, indices])

            pos_indices = []
            neg_indices = []

            # sample anchor, positive, negative for each wrong class
            for cls_id in classes_wrong:
                sampling_time = time.time()
                # cls_indices : anchor, positive indices
                cls_indices = ((M == 1) & (predicts_j[ii] == cls_id)).nonzero()

                # pass if wrong class doesn't exist in the image
                if cls_id not in classes_labels:
                    continue
                else:
                    neg_cls_indices = (labels[ii] == cls_id).nonzero()

                    if neg_cls_indices.size(0) < self.num_patch:
                        continue

                    neg_sampled_indices = [neg_cls_indices[torch.randperm(neg_cls_indices.size(0))[
                                                           :self.num_patch]].squeeze()] * cls_indices.size(0)
                    neg_sampled_indices = torch.cat(neg_sampled_indices, dim=0)

                pos_indices.append(cls_indices)
                neg_indices.append(neg_sampled_indices)

            if not pos_indices:
                continue
            pos_indices = torch.cat(pos_indices, dim=0)
            neg_indices = torch.cat(neg_indices, dim=0)

            # anchor from query feature
            X_pos_q.append(feats_q[ii, pos_indices, :])
            # positive from key feature
            X_pos_k.append(feats_k[ii, pos_indices, :])
            # Negative from query feature
            X_neg.append(feats_q[ii, neg_indices, :].view(pos_indices.size(0), self.num_patch, C))

        if not X_pos_q:
            return None, None, None
        # X_pos_q, X_pos_k shape : (num_patch, 1, C)
        # X_neg shape : (num_patch, negative_size, C)
        X_pos_q = torch.cat(X_pos_q, dim=0)
        X_pos_k = torch.cat(X_pos_k, dim=0)
        X_neg = torch.cat(X_neg, dim=0)

        if X_pos_q.shape[0] > B * self.max_samples:
            indices = torch.randperm(X_pos_q.size(0))[:B * self.max_samples]
            X_pos_q = X_pos_q[indices, :, :]
            X_pos_k = X_pos_k[indices, :, :]
            X_neg = X_neg[indices, :, :]

        return X_pos_q, X_pos_k, X_neg

    def forward(self, feats_q, feats_k, predicts, predicts_j, labels):
        B, C, H, W = feats_q.shape

        # reshape the labels and predictions to feature map's size
        labels = self.reshape_map(labels, (H, W))
        predicts = self.reshape_map(predicts, (H, W))
        predicts_j = self.reshape_map(predicts_j, (H, W))

        # calculate Correction map
        correct_maps = torch.ones_like(predicts, device=feats_q[0].device)
        correct_maps[predicts == predicts_j] = 0
        correct_maps[labels == 255] = 0
        correct_maps[predicts != labels] = 0
        correct_maps = correct_maps.flatten(1, 2)

        predicts = predicts.flatten(1, 2)
        predicts_j = predicts_j.flatten(1, 2)
        labels = labels.flatten(1, 2)

        feats_k = feats_k.detach()

        feats_q_reshape = feats_q.permute(0, 2, 3, 1).flatten(1, 2)
        feats_k_reshape = feats_k.permute(0, 2, 3, 1).flatten(1, 2)

        # Sample the anchor and positives, negatives
        patches_q, patches_k, patches_neg = self.Disentangle_Sampler(correct_maps, feats_q_reshape, feats_k_reshape,
                                                                     predicts, predicts_j, labels)

        if patches_q is None:
            loss = torch.FloatTensor([0]).cuda()
            return loss

        loss = self._contrastive(patches_q, patches_k, patches_neg)

        return loss

ce_weight = None
criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean', ignore_index=-1).cuda()
criterion_CA = nn.MSELoss()
criterion_CWCL = Class_PixelNCELoss(numclass=6)
criterion_SDCL = Disentangle_Contrast(numclass=6)




def CWCL_SDCL(embed_q_, embed_k_, gts, predict, predict_j, qd_arr, use_cwcl=False, use_sdcl=False):
    return_loss = []
    cwcl = torch.FloatTensor([0]).cuda()
    sdcl = torch.FloatTensor([0]).cuda()

    for N, f_maps in enumerate(zip(embed_q_, embed_k_)):
        embed_q, embed_k = f_maps
    # for N, f_maps in enumerate(zip(qd_arr, kd_arr)):
        # feat_q, feat_k = f_maps
        # projection = getattr(self, 'ProjectionHead_cls_%d' % N)
        #
        # embed_q = projection(feat_q)
        # embed_k = projection(feat_k)

        if use_cwcl:
            loss_cw = criterion_CWCL(embed_q, embed_k, gts, predict)
            cwcl = cwcl + loss_cw.mean()

        if use_sdcl:
            loss_sd = criterion_SDCL(embed_q, embed_k, predict, predict_j, gts)
            sdcl = sdcl + loss_sd.mean()

    if use_cwcl:
        cwcl = cwcl / len(qd_arr)
        return_loss.append(cwcl)
    if use_sdcl:
        sdcl = sdcl / len(qd_arr)
        return_loss.append(sdcl)

    return  return_loss
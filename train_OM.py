import logging
import pprint
import argparse
import random
import timeit
import numpy as np
import pickle
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import model_zoo
from lib.utils.utils import *
from lib.data.WHU_strong import WHU, WHU_augstrong_cutmix
from lib.data.OM import OM, OM_augstrong_cutmix
from util.loss import distillation_loss, OhemCrossEntropy
from segmentation import UNet_CL, region_memory, computer_region_center, update_region_memory, region_contrast_loss, computer_region_center_SACL
from sklearn.model_selection import KFold
import copy
from augmentation.tsrm import TRM
from lib.model.deeplabv3plus import DeepLabV3Plus

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument('--cfg',
                        default='./experiments/Potsdam2Vaihingen/same_mode.yaml',
                        # default = './experiments/potsdam/SEMI_contast_cutmix.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--random-seed",
                        type=int,
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=0)
    parser.add_argument("--num_classes",
                        type=int,
                        default=3)
    parser.add_argument("--model",
                        type=str,
                        default='deeplabv2_unimatch')
    parser.add_argument('--crop_size', type=int, default=768,
                        help='training crop size')
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument("--msc",
                        type=bool,
                        default=False)
    parser.add_argument("--lr",
                        type=float,
                        default=3e-4)
    parser.add_argument("--epoch",
                        type=int,
                        default=200)
    parser.add_argument("--contrast_weight_l",
                        type=float,
                        default=0.1)
    parser.add_argument("--contrast_weight_u",
                        type=float,
                        default=0.01)
    parser.add_argument("--EMA_weight",
                        type=float,
                        default=0.999)
    parser.add_argument("--use_best_model",
                        default=False)
    parser.add_argument("--rootoutput",
                        default='./outputs_2026/OM_DP50_sensitive_newaug1-5')
    args = parser.parse_args()

    return args


start = timeit.default_timer()
args = get_arguments()
criterion = nn.BCELoss()


def main():
    if args.random_seed > 0:
        print('Seeding with', args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)


    # init cuda
    cudnn.benchmark = True  # TODO:設置False，判斷加速時長设置
    cudnn.deterministic = False  # TODO:设置True，避免波动，相当于设置CUDNN.BENCHMARK为False
    cudnn.enabled = True  # TODO:配合CUDNN.BENCHMARK为True使用

    rootOutputDir = args.rootoutput
    if not os.path.exists(rootOutputDir):
        os.makedirs(rootOutputDir)
        print(f"创建目录: {rootOutputDir}")

    # load data
    input_size = (512, 512)
    batch_size = 4
    num_workers = 8

    prev_best_model_path = None

    train_dataset_source = OM_augstrong_cutmix('/mnt/sda/su8/LY0421/OM_512/paris',
                                               '/mnt/sda/su8/LY0421/OM_512/paris.lst',
                                               downsample_rate=1,
                                               base_size=1024,
                                               crop_size=input_size,
                                               scale=True,
                                               mirror=True,
                                               ignore_label=-1,
                                               scale_factor=16)
    trainloader_source = torch.utils.data.DataLoader(train_dataset_source,
                                                     batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers, pin_memory=True)
    chicago_dataset = OM('/mnt/sda/su8/LY0421/OM_512/chicago',
                               '/mnt/sda/su8/LY0421/OM_512/chicago_test.lst',
                               downsample_rate=1,
                               base_size=1024,
                               crop_size=input_size,
                               scale=False,
                               mirror=False,
                               ignore_label=-1,
                               scale_factor=16)
    testloader_chicago = torch.utils.data.DataLoader(chicago_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=4, pin_memory=True)
    potsdam_dataset = OM('/mnt/sda/su8/LY0421/OM_512/potsdam',
                         '/mnt/sda/su8/LY0421/OM_512/potsdam_test.lst',
                         downsample_rate=1,
                         base_size=1024,
                         crop_size=input_size,
                         scale=False,
                         mirror=False,
                         ignore_label=-1,
                         scale_factor=16)
    testloader_potsdam = torch.utils.data.DataLoader(potsdam_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=4, pin_memory=True)
    prev_best_model_path_potsdam = None
    prev_best_model_path_all = None
    # init Student
    device_ids = [0]
    device = 'cuda:0'
    # model = get_models(args).to(device)
    model = DeepLabV3Plus(num_class=args.num_classes, backbone='resnet50')
    seg_loss = OhemCrossEntropy().to(device)
    # optimizer for segmentation network
    params_list = []
    for module in zip(model.parameters()):
        params_list.append(
            # dict(params=module.parameters(), lr=config.TRAIN.LR)
            dict(params=module, lr=args.lr)
        )

    optimizer = torch.optim.SGD(params_list,
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=0.0005,
                                nesterov=False,  # TODO:设置为True
                                )
    criterion_u_mix = nn.CrossEntropyLoss().cuda()
    optimizer.zero_grad()
    model = nn.DataParallel(model, device_ids=device_ids)
    model.train()

    trainSteps = int((len(train_dataset_source) / batch_size / args.epoch))
    validPerStep = int((len(train_dataset_source) / batch_size / 1))
    beginSTStep = int((len(train_dataset_source) / batch_size / 50))
    againSTStep = int((len(train_dataset_source) / batch_size / 100))
    best_mIoU_S1, best_mIoU_S2, best_mIoU = 0, 0, 0
    best_f1_S1, best_f1_S2 = 0, 0
    best_loss = 100
    epoch = args.epoch
    w_cl = 0.5
    val_loss_all, val_acc_all = [], []
    edge_enhance_para = 1.5 # 1.0-2.0

    for i_epoch in range(1, epoch + 1):
        val_loss, val_num = 0, 0
        RC_memory_w = region_memory(num_class=args.num_classes, device=device)
        RC_memory_s = region_memory(num_class=args.num_classes, device=device)
        for iter, batch in tqdm(enumerate(trainloader_source)):
            images, images_aug1, images_aug2, cutmix1, cutmix2, labels = batch
            images_t, images_aug1_t, images_aug2_t, cutmix1_t, cutmix2_t = \
                Variable(images).to(device), Variable(images_aug1).to(device), Variable(images_aug2).to(
                    device), cutmix1.to(device), cutmix2.to(device)
            i_iter = i_epoch * (len(train_dataset_source) / batch_size) + iter

            lambdaST = adjustLambdaST(i_iter, beginSTStep, againSTStep, 0.3)
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_epoch, epoch, args)
            model.train()

            # Weskly Augmentation, more domain-invariant feature, improve the performance
            labels_t = labels.long().to(device)
            pred_w, pred_fp, fea_CL, pre_CL = model(images_t, need_fp=True)
            loss_sg_w = seg_loss(pred_w, labels_t)
            loss_sg_fp = seg_loss(pred_fp, labels_t)
            loss_sg_cl_w = seg_loss(F.interpolate(pre_CL, size=pred_w.shape[2:], mode="bilinear", align_corners=True), labels_t)

            # Strong Augmentation, more style information, improve the robust
            randconv = TRM(in_channels=3, out_channels=3, kernel_size_list=[1, 3, 5, 7], layer_max=10, gamma_std=1,
                           beta_std=1, mixing=True).to(device)
            images_aug1_t_blur = randconv(images_aug1_t)
            residual_1 = images_aug1_t - images_aug1_t_blur
            images_aug1_t_sharp = images_aug1_t + edge_enhance_para * residual_1
            images_aug1_t_sharp = torch.clamp(images_aug1_t_sharp, 0, 1)

            images_aug2_t_blur = randconv(images_aug2_t)
            residual_2 = images_aug2_t - images_aug2_t_blur
            images_aug2_t_sharp = images_aug2_t + edge_enhance_para * residual_2
            images_aug2_t_sharp = torch.clamp(images_aug2_t_sharp, 0, 1)

            if random.random() > 0.5:
                images_aug2_t_blur[cutmix2_t.unsqueeze(1).expand(images_aug2_t.shape) == 1] = \
                    images_aug1_t_blur[cutmix2_t.unsqueeze(1).expand(images_aug2_t.shape) == 1]
                pred_s, fea_CL_smix, pre_CL_smix = model(images_aug2_t_blur)
                loss_seg_s = seg_loss(pred_s, labels_t)
                loss_sg_cl_s = seg_loss(F.interpolate(pre_CL_smix, size=pred_w.shape[2:], mode="bilinear", align_corners=True), labels_t)

            else:
                images_aug2_t_sharp[cutmix2_t.unsqueeze(1).expand(images_aug2_t.shape) == 1] = \
                    images_aug1_t_sharp[cutmix2_t.unsqueeze(1).expand(images_aug2_t.shape) == 1]
                pred_s, fea_CL_smix, pre_CL_smix = model(images_aug2_t_sharp)
                loss_seg_s = seg_loss(pred_s, labels_t)
                loss_sg_cl_s = seg_loss(F.interpolate(pre_CL_smix, size=pred_w.shape[2:], mode="bilinear", align_corners=True), labels_t)



            # Contrastive learning
            keys_source, vals_source = computer_region_center(fea_CL, pre_CL, num_class=args.num_classes)
            keys_style_aug, vals_style_aug = computer_region_center(fea_CL_smix, pre_CL_smix, num_class=args.num_classes)
            keys_source1, vals_source1, keys_style1, vals_style1 = computer_region_center_SACL(fea_CL, pre_CL,
                                                                                               fea_CL_smix,
                                                                                               pre_CL_smix,
                                                                                               num_class=args.num_classes)

            for class_i in range(args.num_classes):
                RC_memory_w = update_region_memory(RC_memory_w, keys_source, vals_source, class_i)
                RC_memory_s = update_region_memory(RC_memory_s, keys_style_aug, vals_style_aug, class_i)
            if i_epoch == 1:
                loss_w_contrast = w_cl * region_contrast_loss(args.num_classes, keys_source, vals_source,
                                                              RC_memory_w, device)
                loss_w_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_source1, vals_source1,
                                                                     RC_memory_s, device)

                loss_s_contrast1 = w_cl * region_contrast_loss(args.num_classes, keys_style_aug, vals_style_aug,
                                                               RC_memory_s, device)
                loss_s_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_style1, vals_style1,
                                                                     RC_memory_w, device)


            else:
                loss_w_contrast = w_cl * region_contrast_loss(args.num_classes, keys_source, vals_source,
                                                              RC_memory_use_w, device)
                loss_w_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_source1, vals_source1,
                                                                     RC_memory_use_s, device)


                loss_s_contrast1 = w_cl * region_contrast_loss(args.num_classes, keys_style_aug, vals_style_aug,
                                                               RC_memory_use_s, device)
                loss_s_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_style1, vals_style1,
                                                                     RC_memory_use_w, device)
            w_c = 0.1
            loss = ( loss_sg_w + loss_sg_cl_w + loss_seg_s + loss_sg_cl_s)  + \
                   w_c * ( loss_w_contrast + loss_s_contrast1) + \
                   0.5 * w_c * ( loss_w_contrast_cross1 + loss_s_contrast_cross1) + \
                   0.5 * loss_sg_fp
            loss.backward()
            optimizer.step()


        RC_memory_use_w = RC_memory_w
        RC_memory_use_s = RC_memory_s
        print('evaluate epoch', i_epoch)
        mean_IoU, IoU_array_S2, PA, f1_S2, f1_Array_s2 = validate(args, testloader_chicago, model, CL=True)
        mean_IoU_potsdam, IoU_array_potsdam, PA, f1_potsdam, f1_mean_potsdam = validate(args, testloader_potsdam, model,
                                                                                        CL=True)
        MIOU = (IoU_array_S2[1] + IoU_array_S2[-1]) / 2.
        MIOU_pots = (IoU_array_potsdam[1] + IoU_array_potsdam[-1]) / 2.
        MIOU_str = f"{MIOU:.4f}"
        MIOU_potsdam = f"{MIOU_pots:.4f}"
        MIOU_all = (MIOU_pots + MIOU) / 2.
        MIOU_all_Str = f"{MIOU_all:.4f}"

        if MIOU_all > best_mIoU:
            # 删除上一个最优模型
            if prev_best_model_path_all and os.path.exists(prev_best_model_path_all):
                os.remove(prev_best_model_path_all)
                print(f"已删除旧模型: {prev_best_model_path_all}")

            # 保存当前最优模型
            save_path_all = os.path.join(rootOutputDir, f"best_all_epoch{i_epoch}_miou{MIOU_all_Str}.pth")
            torch.save(model.module.state_dict(), save_path_all)
            print(f"保存新最优模型: {save_path_all}")

            # 更新记录
            best_mIoU = MIOU_all
            prev_best_model_path_all = save_path_all

        if MIOU_pots > best_mIoU_S1:
            # 删除上一个最优模型
            if prev_best_model_path_potsdam and os.path.exists(prev_best_model_path_potsdam):
                os.remove(prev_best_model_path_potsdam)
                print(f"已删除旧potsdam模型: {prev_best_model_path_potsdam}")

            # 保存当前最优模型
            save_path_potsdam = os.path.join(rootOutputDir, f"best_potsdam_epoch{i_epoch}_miou{MIOU_potsdam}.pth")
            torch.save(model.module.state_dict(), save_path_potsdam)
            print(f"保存potsdam新最优模型: {save_path_potsdam}")

            # 更新记录
            best_mIoU_S1 = MIOU_pots
            prev_best_model_path_potsdam = save_path_potsdam

        if MIOU > best_mIoU_S2:
            # 删除上一个最优模型
            if prev_best_model_path and os.path.exists(prev_best_model_path):
                os.remove(prev_best_model_path)
                print(f"已删除旧chicago模型: {prev_best_model_path}")

            # 保存当前最优模型
            save_path = os.path.join(rootOutputDir, f"best_chicago_epoch{i_epoch}_miou{MIOU_str}.pth")
            torch.save(model.module.state_dict(), save_path)
            print(f"保存chicago新最优模型: {save_path}")

            # 更新记录
            best_mIoU_S2 = MIOU
            prev_best_model_path = save_path

        print('chicago', 'miou', IoU_array_S2[1], IoU_array_S2[-1], MIOU)
        print('potsdam', 'miou', IoU_array_potsdam[1], IoU_array_potsdam[-1], MIOU_pots)
        model.train()


# --- 创建增强模块 ---
# class TextureEnhance(nn.Module):
#     def __init__(self, kernel_size=3, scale=1.0):
#         super(TextureEnhance, self).__init__()
#         self.randconv = RandConv(kernel_size)
#         self.scale = scale  # 控制残差信号强度
#
#     def forward(self, x):
#         x_blur = self.randconv(x)  # RandConv 模糊图
#         residual = x - x_blur  # 提取纹理残差
#         enhanced = x + self.scale * residual  # 加回原图
#         enhanced = torch.clamp(enhanced, 0, 1)  # 防止越界
#         return enhanced





if __name__ == '__main__':
    main()




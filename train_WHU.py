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
                        default=2)
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
                        default=300)
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
                        default='./outputs/WH101_sensitive_newaug1-25')
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
    prev_best_model_path_S1 = None
    prev_best_model_path_S2 = None
    train_dataset_source = WHU_augstrong_cutmix('/mnt/sda/su8/LY0421/WH/aerial',
                                                '/mnt/sda/su8/LY0421/WH/aerial/train.lst',
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

    test_dataset_S1 = WHU('/mnt/sda/su8/LY0421/WH/Satellite1',
                          '/mnt/sda/su8/LY0421/WH/Satellite1/all.lst',
                          downsample_rate=1,
                          base_size=1024,
                          crop_size=input_size,
                          scale=False,
                          mirror=False,
                          ignore_label=-1,
                          scale_factor=16)
    testloader_S1 = torch.utils.data.DataLoader(
        test_dataset_S1,
        batch_size=batch_size,  # batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    test_dataset_S2 = WHU('/mnt/sda/su8/LY0421/WH/Satellite2/crop',
                          '/mnt/sda/su8/LY0421/WH/Satellite2/crop/test.lst',
                          downsample_rate=1,
                          base_size=1024,
                          crop_size=input_size,
                          scale=False,
                          mirror=False,
                          ignore_label=-1,
                          scale_factor=16)
    testloader_S2 = torch.utils.data.DataLoader(
        test_dataset_S2,
        batch_size=batch_size,  # batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    # init Student
    device_ids = [4]
    device = 'cuda:4'
    # model = get_models(args).to(device)
    model = DeepLabV3Plus(num_class=args.num_classes, backbone='resnet101')
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
    best_mIoU_S1, best_mIoU_S2 = 0, 0
    best_mIoU = 0
    best_f1_S1, best_f1_S2 = 0, 0
    best_loss = 100
    epoch = args.epoch
    w_cl = 0.5
    val_loss_all, val_acc_all = [], []
    edge_enhance_para = 1.25
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

            num = images_aug1_t.shape[0]
            preds, fea_CL_smix, pre_CL_smix = model(torch.cat([images_aug2_t_sharp, images_aug2_t_blur]))
            pred_s = preds[:num]
            pred_mix = preds[num:]
            loss_seg_s = seg_loss(pred_s, labels_t)
            loss_seg_mix = seg_loss(pred_mix, labels_t)
            loss_sg_cl_s = seg_loss(
                F.interpolate(pre_CL_smix[:num], size=pred_w.shape[2:], mode="bilinear", align_corners=True),
                labels_t) + seg_loss(
                F.interpolate(pre_CL_smix[num:], size=pred_w.shape[2:], mode="bilinear", align_corners=True), labels_t)

            # Contrastive learning
            keys_source, vals_source = computer_region_center(fea_CL, pre_CL, num_class=args.num_classes)
            keys_style_aug, vals_style_aug = computer_region_center(fea_CL_smix, pre_CL_smix,
                                                                    num_class=args.num_classes)
            keys_source1, vals_source1, keys_style1, vals_style1 = computer_region_center_SACL(fea_CL, pre_CL,
                                                                                               fea_CL_smix[:num],
                                                                                               pre_CL_smix[:num],
                                                                                               num_class=args.num_classes)
            keys_source2, vals_source2, keys_style2, vals_style2 = computer_region_center_SACL(fea_CL, pre_CL,
                                                                                               fea_CL_smix[num:],
                                                                                               pre_CL_smix[num:],
                                                                                               num_class=args.num_classes)
            for class_i in range(args.num_classes):
                RC_memory_w = update_region_memory(RC_memory_w, keys_source, vals_source, class_i)
                RC_memory_s = update_region_memory(RC_memory_s, keys_style_aug, vals_style_aug, class_i)
            if i_epoch == 1:
                loss_w_contrast = w_cl * region_contrast_loss(args.num_classes, keys_source, vals_source,
                                                              RC_memory_w, device)
                loss_w_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_source1, vals_source1,
                                                                     RC_memory_s, device)
                loss_w_contrast_cross2 = w_cl * region_contrast_loss(args.num_classes, keys_source2, vals_source2,
                                                                     RC_memory_s, device)

                loss_s_contrast1 = w_cl * region_contrast_loss(args.num_classes, keys_style_aug, vals_style_aug,
                                                               RC_memory_s, device)
                loss_s_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_style1, vals_style1,
                                                                     RC_memory_w, device)
                loss_s_contrast_cross2 = w_cl * region_contrast_loss(args.num_classes, keys_style2, vals_style2,
                                                                     RC_memory_w, device)

            else:
                loss_w_contrast = w_cl * region_contrast_loss(args.num_classes, keys_source, vals_source,
                                                              RC_memory_use_w, device)
                loss_w_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_source1, vals_source1,
                                                                     RC_memory_use_s, device)
                loss_w_contrast_cross2 = w_cl * region_contrast_loss(args.num_classes, keys_source2, vals_source2,
                                                                     RC_memory_use_s, device)

                loss_s_contrast1 = w_cl * region_contrast_loss(args.num_classes, keys_style_aug, vals_style_aug,
                                                               RC_memory_use_s, device)
                loss_s_contrast_cross1 = w_cl * region_contrast_loss(args.num_classes, keys_style1, vals_style1,
                                                                     RC_memory_use_w, device)
                loss_s_contrast_cross2 = w_cl * region_contrast_loss(args.num_classes, keys_style2, vals_style2,
                                                                     RC_memory_use_w, device)

            w_c = 0.1
            loss = (loss_sg_w + loss_sg_cl_w + loss_seg_s + loss_sg_cl_s) + \
                   w_c * (loss_w_contrast + loss_s_contrast1) + \
                   0.5 * w_c * (loss_w_contrast_cross1 + loss_s_contrast_cross1 + loss_w_contrast_cross2+ loss_s_contrast_cross2) + \
                   0.5 * loss_sg_fp
            loss.backward()
            optimizer.step()


        RC_memory_use_w = RC_memory_w
        RC_memory_use_s = RC_memory_s
        print('evaluate epoch:{}'.format(i_epoch))
        mean_IoU_S1, IoU_array_S1, PA_S1, f1_S1, _ = validate(args, testloader_S1, model, CL=True)
        mean_IoU_S2, IoU_array_S2, PA_S2, f1_S2, _ = validate(args, testloader_S2, model, CL=True)
        # mean_IoU, IoU_array_S2, PA, f1_S2, f1_Array_s2 = validate(args, testloader_chicago, model, CL=True)
        MIOU_S2 = IoU_array_S2[-1]
        MIOU_S1 = IoU_array_S1[-1]
        MIOU = (MIOU_S1 + MIOU_S2) / 2.
        MIOU_strS1 = f"{MIOU_S1:.4f}"
        MIOU_strS2 = f"{MIOU_S2:.4f}"
        MIOU_str = f"{MIOU:.4f}"
        if MIOU_S2 > best_mIoU_S2:
            # 删除上一个最优模型
            if prev_best_model_path_S2 and os.path.exists(prev_best_model_path_S2):
                os.remove(prev_best_model_path_S2)
                print(f"已删除旧模型: {prev_best_model_path_S2}")
            # 保存当前最优模型
            save_path_S2 = os.path.join(rootOutputDir, f"best_S2_epoch{i_epoch}_miou{MIOU_strS2}.pth")
            torch.save(model.module.state_dict(), save_path_S2)
            print(f"保存新最优S2模型: {save_path_S2}")
            # 更新记录
            best_mIoU_S2 = MIOU_S2
            prev_best_model_path_S2 = save_path_S2

        if MIOU_S1 > best_mIoU_S1:
            # 删除上一个最优模型
            if prev_best_model_path_S1 and os.path.exists(prev_best_model_path_S1):
                os.remove(prev_best_model_path_S1)
                print(f"已删除旧模型: {prev_best_model_path_S1}")
            # 保存当前最优模型
            save_path_S1 = os.path.join(rootOutputDir, f"best_S1_epoch{i_epoch}_miou{MIOU_strS1}.pth")
            torch.save(model.module.state_dict(), save_path_S1)
            print(f"保存新最优S1模型: {save_path_S1}")
            # 更新记录
            best_mIoU_S1 = MIOU_S1
            prev_best_model_path_S1 = save_path_S1

        if MIOU > best_mIoU:
            # 删除上一个最优模型
            if prev_best_model_path and os.path.exists(prev_best_model_path):
                os.remove(prev_best_model_path)
                print(f"已删除旧模型: {prev_best_model_path}")
            # 保存当前最优模型
            save_path = os.path.join(rootOutputDir, f"best_epoch{i_epoch}_miou{MIOU_str}.pth")
            torch.save(model.module.state_dict(), save_path)
            print(f"保存新最优模型: {save_path}")
            # 更新记录
            best_mIoU = MIOU
            prev_best_model_path = save_path



        print('S1', 'miou', IoU_array_S1[-1])
        print('S2', 'miou', IoU_array_S2[-1])
        model.train()


        # if MIOU > best_mIoU_S2:
        #
        #     torch.save(model.module.state_dict(),
        #                os.path.join(rootOutputDir, 'best_chicago' + str(MIOU) + '.pth'))
        #     best_mIoU_S2 = MIOU
        # # model.load_state_dict(best_model_wts)
        # print('chicago','miou', IoU_array_S2[1], IoU_array_S2[-1], (IoU_array_S2[1] + IoU_array_S2[-1]) / 2.)
        # model.train()




if __name__ == '__main__':
    main()




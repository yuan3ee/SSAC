import logging
import time
import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
# import our stuffs
import lib.utils.distributed as dist
from lib.utils.loss_soft_online import CrossEntropy, OhemCrossEntropy, CrossEntropyST, antiCrossEntropyST


def get_sampler(dataset):
    from lib.utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def one_hot(label, config):
    if not isinstance(label, np.ndarray):
        label = label.detach().cpu().numpy()
    one_hot = np.zeros((label.shape[0], config.DATASET.NUM_CLASSES, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(config.DATASET.NUM_CLASSES):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)


def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output


def compute_argmax_labelmap(pred_remain_ST):
    output = pred_remain_ST.detach().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = np.asarray(np.argmax(output, axis=3), dtype=np.int)
    return output


# def find_good_map_CCT(D_outs, pred_all, pred_all_CCT_list, config):
#     """
#     @D_outs:(bs, 1, H, W)
#     @pred_all:(bs, 6, H, W)，未添加扰动的预测
#     @pred_all_CCT_list:list,元素shape(bs, 6, H, W)，添加了扰动的预测
#     """
#     stList_CCT = list()
#     pixelSum = config.TRAIN.IMAGE_SIZE[0] * config.TRAIN.IMAGE_SIZE[1]
#     for i in range(D_outs.size(0)):
#         stList = list()
#         D_outs_ndarray = D_outs[i].detach().cpu().numpy()
#         if np.amax(D_outs_ndarray) <= config.TRAIN.THRESHOLD_ST:
#             return 0, 0, 0
#         pixelCount = np.sum(D_outs_ndarray > config.TRAIN.THRESHOLD_ST)
#         # print(pixelCount)
#         if (pixelCount / pixelSum) > config.TRAIN.THRESHOLD_SUM:
#             label_CCT = compute_argmax_map(pred_all[i])
#             for pred_all_CCT in pred_all_CCT_list:
#                 stList.append((pred_all_CCT[i], label_CCT))
#             stList_CCT.append(stList)
#     if len(stList_CCT) > 0:
#         print('Above ST-Threshold : ', len(stList_CCT), '/', config.TRAIN.BATCH_SIZE_PER_GPU)
#         pred_sel = torch.Tensor(len(stList_CCT), len(pred_all_CCT_list), pred_all.size(1), pred_all.size(2), pred_all.size(3))
#         label_sel = torch.Tensor(len(stList_CCT), pred_sel.size(-2), pred_sel.size(-1))
#         for m in range(len(stList_CCT)):
#             label_sel[m] = stList_CCT[0][0][1]
#             for n in range(len(stList_CCT[m])):
#                 pred_sel[m][n] = stList_CCT[m][n][0]
#         return pred_sel.cuda(), label_sel.cuda(), len(stList_CCT)
#     else:
#         return 0, 0, 0


# def find_good_maps_FCN(D_outs, pred_all, pred_all_ST, config):
#     """
#     @D_outs:(bs, 1, H, W)
#     @pred_all:(bs, 6, H, W)
#     @pred_all_ST:(bs, 6, H, W)
#     """
#     stList = list()
#     pixelSum = config.TRAIN.IMAGE_SIZE[0] * config.TRAIN.IMAGE_SIZE[1]
#     for i in range(D_outs.size(0)):
#         D_outs_ndarray = D_outs[i][0].detach().cpu().numpy()
#         if np.amax(D_outs_ndarray) <= config.TRAIN.THRESHOLD_ST:
#             return 0, 0, 0
#         pixelCount = np.sum(D_outs_ndarray > config.TRAIN.THRESHOLD_ST)
#         # print(pixelCount)
#         if (pixelCount / pixelSum) > config.TRAIN.THRESHOLD_SUM:
#             stList.append((pred_all_ST[i], compute_argmax_map(pred_all[i])))
#     if len(stList) > 0:
#         print('Above ST-Threshold : ', len(stList), '/', config.TRAIN.BATCH_SIZE_PER_GPU)
#         pred_sel = torch.Tensor(len(stList), pred_all.size(1), pred_all.size(2), pred_all.size(3))
#         label_sel = torch.Tensor(len(stList), pred_sel.size(2), pred_sel.size(3))
#         for j in range(len(stList)):
#             pred_sel[j] = stList[j][0]
#             label_sel[j] = stList[j][1]
#         return pred_sel.cuda(), label_sel.cuda(), len(stList)
#     else:
#         return 0, 0, 0


def adjust_learning_rate(optimizer, i_iter, trainSteps, config):
    lr = lr_poly(config.TRAIN.LR, i_iter, trainSteps)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_D(optimizer, i_iter, trainSteps, config):
    lr = lr_poly(config.TRAIN.LR_D, i_iter, trainSteps)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def lr_poly(base_lr, iter, max_iter, power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))


def loss_calc(pred, label, config):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy(config, ignore_label=config.TRAIN.IGNORE_LABEL).cuda()
    return criterion(pred, label)


def loss_calc_ohem(pred, label, config):
    label = Variable(label.long()).cuda()
    # print('pred', pred.shape)
    # print('label', label.shape)
    criterion = OhemCrossEntropy(config, ignore_label= -1, #config.TRAIN.IGNORE_LABEL,
                                    thres=config.LOSS.OHEMTHRES,
                                    min_kept=config.LOSS.OHEMKEEP).cuda()
    return criterion(pred, label)


def loss_calc_ST(pred, label, D_out_z, config):
    label = Variable(label.long()).cuda()
    pred = pred.cuda()
    criterion = CrossEntropyST(config, D_out_z=D_out_z, ignore_label=config.TRAIN.IGNORE_LABEL).cuda()
    return criterion(pred, label)


def loss_calc_ST_anti(pred, label, D_out_z, config):
    label = Variable(label.long()).cuda()
    pred = pred.cuda()
    criterion = antiCrossEntropyST(config, D_out_z=D_out_z, ignore_label=config.TRAIN.IGNORE_LABEL).cuda()
    return criterion(pred, label)


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def createLogger(args, cfg, phase='train'):
    rootOutputDir = os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET), args.cfg.split('/')[-1].split('.')[0])
    timeStr = time.strftime('%Y-%m-%d-%H-%M')
    logFile = '{}_{}.log'.format(phase, timeStr)
    finalLogFile = os.path.join(rootOutputDir, logFile)
    if not os.path.exists(rootOutputDir):
        os.makedirs(rootOutputDir)
        print('=> creating {}'.format(rootOutputDir))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(finalLogFile), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(rootOutputDir)


def testval(config, test_dataset, testloader, model, model_D=None, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            images_gt = Variable(image).cuda()
            images_gt = ((image - torch.min(image)) / (torch.max(image) - torch.min(image))).cuda()
            pred = test_dataset.multi_scale_inference(config, model, image, scales=config.TEST.SCALE_LIST, flip=config.TEST.FLIP_TEST)

            # if len(border_padding) > 0:
            #     border_padding = border_padding[0]
            #     pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred and model_D:
                # for hung
                interp = torch.nn.Upsample(size=(config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]), mode='bilinear', align_corners=True)
                D_out_z = torch.sigmoid(interp(model_D(F.softmax(pred, dim=1))))

                # pred_cat = torch.cat((F.softmax(pred, dim=1), images_gt), dim=1)
                # _, D_out_z = model_D(pred_cat)

            confusion_matrix += get_confusion_matrix(label, pred, size, config.DATASET.NUM_CLASSES)
            
            if sv_pred:
                sv_path = os.path.join(os.path.join('/home/ly/LY/Semi-RC-segmentation/paper', 'potsdam'), '12-5supervised_ohem_RC0')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name, D_out_z=None)

            # if index % 100 == 0:
            #     logging.info('processing: %d images' % index)
            #     pos = confusion_matrix.sum(1)
            #     res = confusion_matrix.sum(0)
            #     tp = np.diag(confusion_matrix)
            #     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            #     mean_IoU = IoU_array.mean()
            #     logging.info('mIoU: %.4f' % (mean_IoU))
    print(confusion_matrix)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    # pixel_acc = tp.sum()/pos.sum()
    # mean_acc = (tp/np.maximum(1.0, pos)).mean()
    PA = tp.sum() / np.sum(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array#[:-1]
    mean_IoU = IoU_array.mean()
    recall_array = tp / pos  # tp[:-1] / pos[:-1]
    precision_array = tp / res  # tp[:-1] / res[:-1]
    f1_array = (2 * recall_array * precision_array) / (recall_array + precision_array)
    f1 = f1_array.mean()

    return mean_IoU, IoU_array, PA, f1, f1_array  # pixel_acc, mean_acc




def get_confusion_matrix_pseudo(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(output, dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    seg_pred = np.squeeze(seg_pred)

    # print('seg_pred', seg_pred, seg_pred.shape)
    # print('seg_gt', seg_gt, seg_gt.shape)
    ignore_index = seg_gt != ignore
    # ignore_index = seg_pred != ignore
    # print('ignore_index', ignore_index)
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    ignore_index = seg_gt != ignore
    # print('seg_gt', seg_gt,seg_gt.shape)
    # print('seg_pred', seg_pred,seg_pred.shape)
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]


    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def validate_pseudolabel(config, pseudo_label, label):
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        size = label.size()
        label = label.long().cuda()
        try:
            pred = pseudo_label
        except Exception as e:
            pass
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        for i, x in enumerate(pred):
            # x = F.interpolate(
            #     input=x, size=size[-2:],
            #     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            # )
            x = torch.unsqueeze(x, dim=1)
            confusion_matrix[..., i] += get_confusion_matrix_pseudo(
                label,
                x,
                size,
                config.DATASET.NUM_CLASSES
            )

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    print('confusion_matrix', confusion_matrix.shape,confusion_matrix)
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        PAArray = tp / np.sum(confusion_matrix[..., i])
        PA = PAArray.sum()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        IoU_array = IoU_array#[:-1]
        mean_IoU = IoU_array[:-1].mean()
        recall_array = tp / pos  # tp[:-1] / pos[:-1]
        precision_array = tp / res  # tp[:-1] / res[:-1]
        f1_array = (2 * recall_array * precision_array) / (recall_array + precision_array)
        f1 = f1_array.mean()

    return mean_IoU, IoU_array, PA, f1

def validate(config, testloader, model):
    model.eval()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
        # for batch in tqdm(testloader):
            image, label, _, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            try:
                pred = model(image)
                pred = pred[:2]
            except Exception as e:
                print(idx)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES
                )
            del image, label, pred
            # if idx % 10 == 0:
            #     print(idx)

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        PAArray = tp / np.sum(confusion_matrix[..., i])
        PA = PAArray.sum()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        IoU_array = IoU_array#[:-1]
        mean_IoU = IoU_array[:-1].mean()
        recall_array = tp / pos  # tp[:-1] / pos[:-1]
        precision_array = tp / res  # tp[:-1] / res[:-1]
        f1_array = (2 * recall_array * precision_array) / (recall_array + precision_array)
        f1 = f1_array.mean()

        # if dist.get_rank() <= 0:
        #     logging.info('{} {} {} {}'.format(i, IoU_array, mean_IoU, PA))
        #     logging.info('{} {} {}'.format(i, f1_array, f1))

    return mean_IoU, IoU_array, PA, f1



def adjustLambdaST(i_iter, t1 ,t2, af):
    alpha = 0.0
    if i_iter > t1:
        alpha = (i_iter - t1) / (t2 - t1) * af
        if i_iter > t2:
            alpha = af
    return alpha
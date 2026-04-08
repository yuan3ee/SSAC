from pyexpat import features

# from base_model import VGG
import torch.nn as nn
import torch


def region_memory(num_class, device):
    region_dic = {}
    for i in range(num_class):
        region_dic['region_class'+str(i)] = torch.zeros(size=[256,1]).to(device)
    return region_dic

def computer_region_center_SACL_before(fea_source, pred_source, fea_style, pred_style, num_class):
    bs = fea_source.shape[0]
    confidence_source = nn.functional.softmax(pred_source, dim=1)
    confidence_style = nn.functional.softmax(pred_style, dim=1)
    confidence_difference = torch.abs(confidence_source - confidence_style) #直接用置信度差值还是用个指数形式的？
    confidence_difference = confidence_difference.view(bs, num_class, -1).permute(1, 0, 2)

    pred_source = pred_source.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    val_source = torch.unique(pred_source)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea_source = fea_source.squeeze()
    fea_source = fea_source.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]
    # print('fea_source', fea_source.shape, val_source.shape, confidence_difference.shape)
    # print(fea_source[:, pred_source == val_source[0]].shape, confidence_difference[val_source[0],pred_source==val_source[0]].shape)
    # new_fea_source = fea_source[:, pred_source == val_source[0]].mean(1).unsqueeze(0)  # new_fea.shape=[1, 256], 此处为0类别的region center
    new_fea_source = fea_source[:, pred_source == val_source[0]] * confidence_difference[val_source[0],pred_source==val_source[0]]
    new_fea_source = new_fea_source.mean(1).unsqueeze(0)
    for i in val_source[1:]:
        if (i < num_class):
            # class_fea = fea_source[:, pred_source == i].mean(1).unsqueeze(0)
            class_fea = fea_source[:, pred_source == i] * confidence_difference[i, pred_source == i]
            class_fea = class_fea.mean(1).unsqueeze(0)
            new_fea_source = torch.cat((new_fea_source, class_fea), dim=0)
    val_source = torch.tensor([i for i in val_source if i < num_class])
    new_fea_source = nn.functional.normalize(new_fea_source, dim=1)

    pred_style = pred_style.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    val_style = torch.unique(pred_style)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea_style = fea_style.squeeze()
    fea_style = fea_style.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]
    new_fea_style = fea_style[:, pred_source == val_source[0]] * confidence_difference[val_source[0], pred_source==val_source[0]]
    new_fea_style = new_fea_style.mean(1).unsqueeze(0)
    for i in val_style[1:]:
        if (i < num_class):
            # class_fea = fea_source[:, pred_source == i].mean(1).unsqueeze(0)
            class_fea = fea_style[:, pred_source == i] * confidence_difference[i, pred_source == i]
            class_fea = class_fea.mean(1).unsqueeze(0)
            new_fea_style = torch.cat((new_fea_style, class_fea), dim=0)
    val_style = torch.tensor([i for i in val_style if i < num_class])
    new_fea_style = nn.functional.normalize(new_fea_style, dim=1)

    return new_fea_source, val_source, new_fea_style, val_style  # new_fea.shape = [6, 256], v

def computer_region_center_SACL(fea_source, pred_source, fea_style, pred_style, num_class):
    bs = fea_source.shape[0]
    confidence_source = nn.functional.softmax(pred_source, dim=1)

    confidence_style = nn.functional.softmax(pred_style, dim=1)
    confidence_difference = torch.abs(confidence_source - confidence_style) #直接用置信度差值还是用个指数形式的？
    confidence_difference = confidence_difference.view(bs, num_class, -1).permute(1, 0, 2)

    pred_source = pred_source.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    pred_style = pred_style.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    pred_difference = (pred_source - pred_style).bool()
    pred_difference = ~pred_difference
    pred_difference = pred_difference.int()

    val_source = torch.unique(pred_source)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea_source = fea_source.squeeze()
    fea_source = fea_source.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]
    # print('fea_source', fea_source.shape, val_source.shape, confidence_difference.shape)
    # print(fea_source[:, pred_source == val_source[0]].shape, confidence_difference[val_source[0],pred_source==val_source[0]].shape)
    # new_fea_source = fea_source[:, pred_source == val_source[0]].mean(1).unsqueeze(0)  # new_fea.shape=[1, 256], 此处为0类别的region center


    # new_fea_source = fea_source[:, pred_source == val_source[0]] * pred_difference[0] #confidence_difference[val_source[0],pred_source==val_source[0]]
    all_fea_source = fea_source * pred_difference
    new_fea_source = all_fea_source[:, pred_source == val_source[0]]
    new_fea_source = new_fea_source.mean(1).unsqueeze(0)
    for i in val_source[1:]:
        if (i < num_class):
            # class_fea = fea_source[:, pred_source == i].mean(1).unsqueeze(0)
            class_fea = all_fea_source[:, pred_source == i] #confidence_difference[i, pred_source == i]
            class_fea = class_fea.mean(1).unsqueeze(0)
            new_fea_source = torch.cat((new_fea_source, class_fea), dim=0)
    val_source = torch.tensor([i for i in val_source if i < num_class])
    new_fea_source = nn.functional.normalize(new_fea_source, dim=1)

    pred_style = pred_style.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    val_style = torch.unique(pred_style)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea_style = fea_style.squeeze()
    fea_style = fea_style.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]
    all_fea_style = fea_style * pred_difference #confidence_difference[val_source[0], pred_source==val_source[0]]
    new_fea_style = all_fea_style[:, pred_source == val_source[0]]
    new_fea_style = new_fea_style.mean(1).unsqueeze(0)
    for i in val_style[1:]:
        if (i < num_class):
            # class_fea = fea_source[:, pred_source == i].mean(1).unsqueeze(0)
            class_fea = all_fea_style[:, pred_source == i] #confidence_difference[i, pred_source == i]
            class_fea = class_fea.mean(1).unsqueeze(0)
            new_fea_style = torch.cat((new_fea_style, class_fea), dim=0)
    val_style = torch.tensor([i for i in val_style if i < num_class])
    new_fea_style = nn.functional.normalize(new_fea_style, dim=1)

    return new_fea_source, val_source, new_fea_style, val_style  # new_fea.shape = [6, 256], v

def computer_region_center(fea, pred, num_class):
    bs = fea.shape[0]

    pred = pred.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    val = torch.unique(pred)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea = fea.squeeze()
    fea = fea.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]

    new_fea = fea[:, pred == val[0]].mean(1).unsqueeze(0)  # new_fea.shape=[1, 256], 此处为0类别的region center
    for i in val[1:]:
        if (i < num_class):
            class_fea = fea[:, pred == i].mean(1).unsqueeze(0)
            new_fea = torch.cat((new_fea, class_fea), dim=0)
    val = torch.tensor([i for i in val if i < num_class])
    new_fea = nn.functional.normalize(new_fea, dim=1)
    return new_fea, val   # new_fea.shape = [6, 256], v

def update_region_memory(region_dic, keys, vals, cat):
    if cat not in vals:
        return region_dic
    keys = keys[list(vals).index(cat)]
    keys = torch.unsqueeze(keys, dim=-1)
    try:
        region_dic['region_class' + str(cat)] = torch.cat((region_dic['region_class' + str(cat)].cuda(), keys), dim=-1)
    except:
        return region_dic
    return region_dic

criterion = nn.CrossEntropyLoss()

def compute_contrast_loss(l_pos, l_neg, device, temperature=0.2):
    N = l_pos.size(0)
    logits = torch.cat((l_pos, l_neg), dim=1)
    logits /= temperature
    labels = torch.zeros((N,), dtype=torch.long).to(device)
    # print(logits.type, labels.type)
    return criterion(logits, labels)


def region_contrast_loss(num_class, keys, vals, region_memory, device):
    contrast_loss = torch.tensor(0.).to(device)
    for cls_ind in range(num_class):
        if cls_ind in vals:
            # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
            query = keys[list(vals).index(cls_ind)]  # 256,
            l_pos = query.unsqueeze(1) * region_memory['region_class' + str(cls_ind)].clone().detach()
            all_ind = [m for m in range(num_class)]
            l_neg = 0
            tmp = all_ind.copy()
            tmp.remove(cls_ind)
            for cls_ind2 in tmp:
                try:
                    l_neg += query.unsqueeze(1) * region_memory['region_class'+str(cls_ind2)].clone().detach()
                except:
                    new = query.unsqueeze(1) * region_memory['region_class' + str(cls_ind2)].clone().detach()
                    if new.shape[-1] < l_neg.shape[-1]:
                        dims = l_neg.shape[-1] - new.shape[-1]
                        new = torch.cat((new, torch.zeros(size=[256, dims]).cuda()), dim=-1)
                        l_neg += new
                    elif new.shape[-1] > l_neg.shape[-1]:
                        dims = new.shape[-1] - l_neg.shape[-1]
                        l_neg = torch.cat((l_neg, torch.zeros(size=[256, dims]).cuda()), dim=-1)
                        l_neg += new
            contrast_loss += compute_contrast_loss(l_pos, l_neg, device=device)
        else:
            continue
    return contrast_loss



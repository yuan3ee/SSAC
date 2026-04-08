import os
import cv2
import random
import torch
import torchvision
import collections
import logging
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data
from torch.nn import functional as F
from . import augs_TIBA as img_trsform
from .transform_w import crop, hflip, normalize, resize, color_transformation
from copy import deepcopy

colormap_dic = {
    'Satellite2': [[0, 0, 0], [255, 255, 255]],
    'Satellite1': [[0, 0, 0], [255, 255, 255]],
    'Aerial': [[0, 0, 0], [255, 255, 255]],
    'paris': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'postdam': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'berlin': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'chicago': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'tokyo': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'zurich': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
}

def image2label(image, dataset_name):
    colormap = colormap_dic[dataset_name]
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1] * 256 + cm[2])] = i
    image = np.int64(image)
    ix = (image[:, :, 0] * 256 + image[:, :, 1] * 256 + image[:, :, 2])
    image2 = cm2lbl[ix]
    return np.int8(image2)


def label2image(prelabel, dataset_name):
    colormap = colormap_dic[dataset_name]
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h * w, -1)
    image = np.zeros((h * w, 3), dtype="int8")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)

def build_additional_strong_transform():
    # assert cfg.get("strong_aug", False) != False
    # strong_aug_nums = cfg["strong_aug"].get("num_augs", 2)
    # flag_use_rand_num = cfg["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(num_augs=1,
            flag_using_random_num=True)
    return strong_img_aug


def build_basic_transfrom(cfg, split="val", mean=[0.485, 0.456, 0.406]):
    ignore_label = 255
    trs_form = []
    if split != "val":
        if cfg.get("rand_resize", False):
            trs_form.append(img_trsform.Resize(cfg.get("resize_base_size", [1024, 2048]), cfg["rand_resize"]))

        if cfg.get("flip", False):
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))

        # crop also sometime for cityscape
        if cfg.get("crop", False):
            crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
            trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))

    return img_trsform.Compose(trs_form)


class OM(data.Dataset):
    def __init__(self, root, list_path, downsample_rate=1, num_samples=None, label_flag='remain', base_size=1024, max_iters=None, crop_size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True, ignore_label=-1, scale_factor=16):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = self.read_files()
        if num_samples:
            if label_flag == 'remain':
                self.files = self.files[num_samples:]
            elif label_flag == 'gt':
                logging.info("use {} data".format(num_samples))
                self.files = self.files[:num_samples]
            else:
                raise ValueError('please input correct label_flag')
        else:
            logging.info("use all data from {}".format(osp.join(self.root, self.list_path)))
        self.scale_factor = scale_factor
        self.base_size = base_size
        self.downsample_rate = downsample_rate


    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_ids:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": osp.join(self.root, image_path),
                    "label": osp.join(self.root, label_path),
                    "name": name,
                })
        else:
            for item in self.img_ids:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": osp.join(self.root, image_path),
                    "label": osp.join(self.root, label_path),
                    "name": name,
                    "weight": 1
                })
        return files


    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue)

        return pad_image


    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label


    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int32(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int32(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int32(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label


    def gen_sample(self, image, label, scale=True, is_mirror=True):
        if scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label


    def random_brightness(self, img):
        # if not config.TRAIN.RANDOM_BRIGHTNESS:
        #     return img
        if not False:
            return img
        if random.random() < 0.5:
            return img
        # self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
        self.shift_value = 10
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image


    def label_transform(self, label):
        return np.array(label).astype('int32')


    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        self.num_classes = config.DATASET.NUM_CLASSES
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int32(self.crop_size[0] * 1.0)
        stride_w = np.int32(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int32(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int32(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred


    def inference(self, config, model, image, flip=False):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()


    def save_pred(self, preds, sv_path, name, D_out_z=None):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        if D_out_z:
            D_out_z_imgs = D_out_z.detach().cpu().numpy()
        for i in range(preds.shape[0]):
            pred = preds[i]
            if D_out_z:
                D_out_z_img = D_out_z_imgs[i][0]
            # print(pred.shape)  # (512, 512)
            predPath = sv_path + '/S-out/{}.npy'.format(name[i])
            predDPath = sv_path + '/D-out/{}.npy'.format(name[i])
            np.save(str(predPath), pred)
            if D_out_z:
                np.save(str(predDPath), D_out_z_img)
            # save_img.save(os.path.join(sv_path, name[i]+'.png'))


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles['img'])
        label = np.array(Image.open(datafiles['label']))
        label = image2label(label, 'paris')
        # label = torch.from_numpy(label).permute(2, 0, 1)[0,:,:].squeeze()
        image, label = normalize(image, label)
        size = image.shape
        name = datafiles["name"]


        # label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        # print('image', image.shape, label.shape)
        return image, label, np.array(size), name, index


    def __len__(self):
        return len(self.files)

class OM_augstrong_cutmix(data.Dataset):
    def __init__(self, root, list_path, downsample_rate=1, num_samples=None, label_flag='remain', base_size=1024, max_iters=None, crop_size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True, ignore_label=-1, scale_factor=16):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = self.read_files()
        if num_samples:
            if label_flag == 'remain':
                self.files = self.files[num_samples:]
            elif label_flag == 'gt':
                logging.info("use {} data".format(num_samples))
                self.files = self.files[:num_samples]
            else:
                raise ValueError('please input correct label_flag')
        else:
            logging.info("use all data from {}".format(osp.join(self.root, self.list_path)))
        self.scale_factor = scale_factor
        self.base_size = base_size
        self.downsample_rate = downsample_rate
        # self.weak_aug = build_basic_transfrom(cfg, split=split, mean=mean)
        self.strong_aug = build_additional_strong_transform()
        # self.weak_aug = build_basic_transfrom(cfg, split=split, mean=mean)

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_ids:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": osp.join(self.root, image_path[0]),
                    "name": name,
                })
        else:
            for item in self.img_ids:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": osp.join(self.root, image_path),
                    "label": osp.join(self.root, label_path),
                    "name": name,
                    "weight": 1
                })
        return files


    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue)

        return pad_image


    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label


    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label


    def gen_sample(self, image, label, scale=True, is_mirror=True):
        if scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label


    def random_brightness(self, img):
        # if not config.TRAIN.RANDOM_BRIGHTNESS:
        #     return img
        if not False:
            return img
        if random.random() < 0.5:
            return img
        # self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
        self.shift_value = 10
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        # print('*****',image.shape)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image


    def label_transform(self, label):
        return np.array(label).astype('int32')


    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        self.num_classes = config.DATASET.NUM_CLASSES
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def multi_scale_inference_pseudo(self, config, model, image, discriminator,scales=[1], flip=False):
        self.num_classes = config.DATASET.NUM_CLASSES
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred


    def inference(self, config, model, image, flip=False):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()


    def save_pred(self, preds, sv_path, name, D_out_z=None):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        if D_out_z:
            D_out_z_imgs = D_out_z.detach().cpu().numpy()
        for i in range(preds.shape[0]):
            pred = preds[i]
            if D_out_z:
                D_out_z_img = D_out_z_imgs[i][0]
            # print(pred.shape)  # (512, 512)
            predPath = sv_path + '/S-out/{}.npy'.format(name[i])
            predDPath = sv_path + '/D-out/{}.npy'.format(name[i])
            np.save(str(predPath), pred)
            if D_out_z:
                np.save(str(predDPath), D_out_z_img)
            # save_img.save(os.path.join(sv_path, name[i]+'.png'))

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles['img'])

        # image_w, image_s1, image_s2 = deepcopy(image), deepcopy(image), deepcopy(image)
        # to_tensor = torchvision.transforms.ToTensor()  # 只做 [0,255] -> [0,1] tensor
        # image_w = to_tensor(image_w)
        # return image_w, image_w, image_w, image_w, image_w, image_w, datafiles['img']

        label = np.array(Image.open(datafiles['label']))
        label = image2label(label, 'paris')
        label = Image.fromarray(label)
        image, label = resize(image, label, (0.5, 2.0))
        ignore_value = 255
        image, label = crop(image, label, 512, ignore_value)
        image, label = hflip(image, label, p=0.5)
        label = np.array(label)
        label = np.expand_dims(label, axis=-1)
        # label[label == 255] = 1

        label = torch.from_numpy(label).permute(2, 0, 1)[0,:,:].squeeze()
        image_w, image_s1, image_s2 = deepcopy(image), deepcopy(image), deepcopy(image)
        image_s1 = color_transformation(image_s1)
        image_s2 = color_transformation(image_s2)
        cutmix_box1 = obtain_cutmix_box(image_s1.size[0], p=0.5)
        cutmix_box2 = obtain_cutmix_box(image_s2.size[0], p=0.5)


        return normalize(image_w), normalize(image_s1), normalize(image_s2), cutmix_box1, cutmix_box2, label





    def __len__(self):
        return len(self.files)

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
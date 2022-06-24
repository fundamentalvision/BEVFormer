import numpy as np
import torch
import math
import cv2

import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from data import convert_mask, convert_one_hot, MAX_OBJECT

class Compose(object):
    """
    Combine several transformation in a serial manner
    """

    def __init__(self, transform=[]):
        self.transforms = transform

    def __call__(self, imgs, annos, use_image):

        for m in self.transforms:
            imgs, annos = m(imgs, annos, use_image)

        return imgs, annos

class Transpose(object):

    """
    transpose the image and mask
    """

    def __call__(self, imgs, annos, use_image):

        H, W, _ = imgs[0].shape
        if H < W:
            return imgs, annos
        else:
            timgs = [np.transpose(img, [1, 0, 2]) for img in imgs]
            tannos = [np.transpose(anno, [1, 0, 2]) for anno in annos]

            return timgs, tannos

class RandomAffine(object):

    """
    Affine Transformation to each frame
    """

    def __call__(self, imgs, annos, use_image):

        if use_image:
            seq = iaa.Sequential([
                iaa.Crop(percent=(0.0, 0.1), keep_size=True),
                iaa.Affine(scale=(0.9, 1.1), shear=(-15, 15), rotate=(-25, 25))
            ])
        else:
            seq = iaa.Sequential([
                iaa.Crop(percent=(0.0, 0.1), keep_size=True),
                iaa.Affine(scale=(0.95, 1.05), shear=(-10, 10), rotate=(-15, 15))
            ])

            seq = seq.to_deterministic()

        num = len(imgs)
        for idx in range(1, num):
            img = imgs[idx]
            anno = annos[idx]
            max_obj = anno.shape[2]-1

            anno = convert_one_hot(anno, max_obj)
            segmap = SegmentationMapsOnImage(anno, shape=img.shape)
            img_aug, segmap_aug = seq(image=img, segmentation_maps=segmap)
            imgs[idx] = img_aug
            annos[idx] = convert_mask(segmap_aug.get_arr(), max_obj)

        return imgs, annos

class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, imgs, annos, use_image):
        v = np.random.uniform(-self.delta, self.delta)
        for id, img in enumerate(imgs):
            imgs[id] += v

        return imgs, annos


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, imgs, annos, use_image):
        v = np.random.uniform(self.lower, self.upper)
        for id, img in enumerate(imgs):
            imgs[id] *= v

        return imgs, annos


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, use_image):

        v = random.randint(0, 1)
        if v == 0:
            return imgs, annos

        sample = imgs[0]
        h, w = sample.shape[:2]

        for id, img in enumerate(imgs):
            imgs[id] = img[:, ::-1, :]

        for id, anno in enumerate(annos):
            annos[id] = anno[:, ::-1, :]

        return imgs, annos

class ToFloat(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, use_image):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)

        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)

        return imgs, annos

class Rescale(object):

    """
    rescale the size of image and masks
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos, use_image):

        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, anno.shape[2]), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno
            annos[id] = canvas

        return imgs, annos

class Stack(object):

    """
    stack adjacent frames into input tensors
    """

    def __call__(self, imgs, annos, use_image):

        num_img = len(imgs)
        num_anno = len(annos)

        h, w, = imgs[0].shape[:2]

        assert num_img == num_anno
        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)

        return img_stack, anno_stack

class ToTensor(object):

    """
    convert to torch.Tensor
    """

    def __call__(self, imgs, annos, use_image):

        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()

        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.permute(0, 3, 1, 2).contiguous()

        return imgs, annos

class Normalize(object):

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)

    def __call__(self, imgs, annos, use_image):

        for id, img in enumerate(imgs):
            imgs[id] = (img / 255.0 - self.mean) / self.std

        return imgs, annos

class ReverseClip(object):

    def __call__(self, imgs, annos, use_image):

        return imgs[::-1], annos[::-1]

class SampleObject(object):

    def __init__(self, num):
        self.num = num

    def __call__(self, imgs, annos, use_image):

        max_obj = annos[0].shape[2] - 1
        num_obj = 0

        valid = np.sum(annos[0].reshape([-1, 1+max_obj]), axis=0) > 0
        annos = [anno[:, :, valid] for anno in annos]

        num_obj = annos[0].shape[2] - 1
        # while num_obj < max_obj and np.sum(annos[0][:, :, num_obj+1]) > 0:
        #     num_obj += 1

        if num_obj <= self.num:
            for idx, anno in enumerate(annos):
                new_anno = np.zeros(anno.shape[:2]+(self.num+1,))
                new_anno[:, :, :num_obj+1] = anno
                annos[idx] = new_anno
        else:
            sampled_idx = random.sample(range(1, num_obj+1), self.num)
            sampled_idx.sort()
            for idx, anno in enumerate(annos):
                new_anno = np.zeros(anno.shape[:2]+(self.num+1,))
                new_anno[:, :, 1:self.num+1] = anno[:, :, sampled_idx]
                new_anno[:, :, 0] = anno[:, :, 0]
                annos[idx] = new_anno

        return imgs, annos

class TrainTransform(object):

    def __init__(self, size, use_image=False):
        self.transform = Compose([
            Transpose(),
            # SampleObject(num=MAX_OBJECT),
            # RandomAffine(),
            ToFloat(),
            # RandomContrast(),
            AdditiveNoise(),
            # RandomMirror(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),

        ])

    def __call__(self, imgs, annos, use_image):
        return self.transform(imgs, annos, use_image)


class TestTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            ToFloat(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),

        ])

    def __call__(self, imgs, annos, use_image):
        return self.transform(imgs, annos, use_image)


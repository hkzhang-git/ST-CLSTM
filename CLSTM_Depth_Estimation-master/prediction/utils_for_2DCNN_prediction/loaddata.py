from torch.utils.data import Dataset, DataLoader
# from prediction_demo.utils_for_2DCNN_prediction.nyu_transform import *
import os
import json
import torch
import numpy as np
from PIL import Image
import collections

try:
    import accimage
except ImportError:
    accimage = None


def load_annotation_data(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dict_dir, root_dir, transform=None):
        self.data_dict = load_annotation_data(dict_dir)
        self.transform = transform
        self.root_dir = root_dir

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.data_dict[idx]['data_path'])
        depth_name = os.path.join(self.root_dir, self.data_dict[idx]['gt_path'])

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['depth'], sample['depth_scaled']

    def __len__(self):
        return len(self.data_dict)


def getTestingData(batch_size=64, dict_dir=None, root_dir=None, num_workers=4):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}


    transformed_testing = depthDataset(dict_dir=dict_dir,
                                       root_dir=root_dir,
                                       transform=Compose([
                                           ReScale(240),
                                           Crop([8, 8, 312, 236], [152, 114]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=False)

    return dataloader_testing


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ReScale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.changeScale(image, self.size)
        depth_scaled = self.changeScale(depth, self.size, Image.NEAREST)

        return {'image': image, 'depth': depth, 'depth_scaled': depth_scaled}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class Crop(object):
    def __init__(self, crop_position, depth_size):
        self.crop_position = crop_position
        self.depth_size = depth_size

    def __call__(self, sample):
        image, depth, depth_scaled = sample['image'], sample['depth'], sample['depth_scaled']
        x1, y1, x2, y2 = self.crop_position[0], self.crop_position[1], self.crop_position[2], self.crop_position[3]
        image = image.crop((x1, y1, x2, y2))
        ow, oh = self.depth_size[0], self.depth_size[1]
        depth_scaled = depth_scaled.crop((x1, y1, x2, y2)).resize((ow, oh))

        return {'image': image, 'depth': depth, 'depth_scaled': depth_scaled}


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        self.is_test = None

    def __call__(self, sample):
        image, depth, depth_scaled = sample['image'], sample['depth'], sample['depth_scaled']
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float() / 6000.0
        depth_scaled = self.to_tensor(depth_scaled).float() / 6000.0
        return {'image': image, 'depth': depth, 'depth_scaled': depth_scaled}

    def to_tensor(self, pic):

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, depth, depth_scaled = sample['image'], sample['depth'], sample['depth_scaled']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'depth': depth, 'depth_scaled': depth_scaled}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

import math
import torch
import torch.nn.functional as F


class metric_list(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, prediction, gt):
        for t in self.transforms:
            t(prediction, gt)

    def loss_get(self):
        results = []
        for t in self.transforms:
            acc = t.loss_get()
            info = {
                'metric': t.metric_name,
                'acc': acc
            }
            results.append(info)
        return results

    def reset(self):
        for t in self.transforms:
            t.reset()


# mean Square error
class RMS(object):
    def __init__(self, metric_name='RMS'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name

    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)

        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        diff = (up_prediction[:, :, 28:455, 24:585] - gt[:, :, 44:471, 40:601])
        square_diff = diff * diff
        rms_sum = float(square_diff.sum())
        b, c, h, w = square_diff.shape
        self.loss += rms_sum
        self.pixel_num += float(b*c*h*w)

    def loss_get(self, frac=4):
        return round(math.sqrt(self.loss / self.pixel_num), frac)

    def reset(self):
        self.loss = 0
        self.pixel_num = 0
        self.scaled_loss = 0
        self.scaled_pixel_num = 0


# average relative error
class REL(object):
    def __init__(self, metric_name='REL'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name

    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)

        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        abs_diff = (up_prediction[:, :, 28:455, 24:585] - gt[:, :, 44:471, 40:601]).abs()
        absrel_sum = float((abs_diff / gt[:, :, 44:471, 40:601]).sum())
        b, c, h, w = abs_diff.shape
        self.loss += absrel_sum
        self.pixel_num += float(b*c*h*w)

    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)

    def reset(self):
        self.loss = 0
        self.pixel_num = 0


# Mean log 10 error (log10)
class log10(object):
    def __init__(self, metric_name='log10'):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name

    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)

        up_prediction = F.upsample(prediction, [h_p*4, w_p*4], mode='bilinear', align_corners=True)
        log10_diff = (torch.log10(up_prediction[:, :, 28:455, 24:585]) - torch.log10(gt[:, :, 44:471, 40:601])).abs()
        log10_sum = float(log10_diff.sum())
        b, c, h, w = log10_diff.shape
        self.loss += log10_sum
        self.pixel_num += float(b*c*h*w)

    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)

    def reset(self):
        self.loss = 0
        self.pixel_num = 0


# thresholded accuracy (deta)
class deta(object):
    def __init__(self, metric_name='deta', threshold=1.25):
        self.loss = 0
        self.pixel_num = 0
        self.metric_name = metric_name
        self.threshold = threshold

    def __call__(self, prediction, gt):
        h_p, w_p = prediction.size(2), prediction.size(3)

        up_prediction = F.upsample(prediction, [h_p * 4, w_p * 4], mode='bilinear', align_corners=True)
        up_prediction_region = up_prediction[:, :, 28:455, 24:585]
        gt_region = gt[:, :, 44:471, 40:601]

        deta_matrix = torch.cat((up_prediction_region / gt_region, gt_region/up_prediction_region), 1).max(1)[0]

        b, c, h, w = gt_region.shape
        self.loss += float((deta_matrix < self.threshold).sum())
        self.pixel_num += float(b * c * h * w)

    def loss_get(self, frac=4):
        return round(self.loss / self.pixel_num, frac)

    def reset(self):
        self.loss = 0
        self.pixel_num = 0




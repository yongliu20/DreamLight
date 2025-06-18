'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''
import cv2 
import numpy as np 
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage


def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor, levels=5):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat, levels)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat, levels)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


def bytes2array(byte_data):
    arr = np.asarray(bytearray(byte_data), dtype="uint8")
    arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return arr


def masks_specified_color_alignment(img_org, img_trg, cal_region, plotting=False):
    # breakpoint()
    if cal_region is not None:
        if len(cal_region.shape)==3:
            cal_region = cal_region[:,:,0]
        temp = img_org[cal_region]
        std_org, mean_org = np.std(temp, axis=0), np.mean(temp, axis=0)
        temp = img_trg[cal_region]
        std_trg, mean_trg = np.std(temp, axis=0), np.mean(temp, axis=0)
    else:
        std_org, mean_org = np.std(img_org, axis=(0,1)), np.mean(img_org, axis=(0,1))
        std_trg, mean_trg = np.std(img_trg, axis=(0,1)), np.mean(img_trg, axis=(0,1))
    
    std_org = std_org[np.newaxis, np.newaxis]
    mean_org = mean_org[np.newaxis, np.newaxis]
    std_trg = std_trg[np.newaxis, np.newaxis]
    mean_trg = mean_trg[np.newaxis, np.newaxis]

    img = img_trg.astype(np.float32) 
    alpha=1.
    norm_out = (img-mean_trg)/std_trg * std_org + mean_org*alpha

    norm_out = np.clip(norm_out, 0, 255)
    norm_out = norm_out.astype(np.uint8)
    
    if cal_region is not None:
        cal_region = (cal_region*255).astype(np.uint8)
        cal_region = 255-cv2.GaussianBlur(cal_region, (5,5), 3)
        cal_region = 255-cv2.GaussianBlur(cal_region, (5,5), 3)
        cal_region = cal_region.astype(np.float32)/255.

        norm_out = (cal_region[...,np.newaxis]) * norm_out.astype(np.float32) + (1.-cal_region[...,np.newaxis]) * img_org.astype(np.float32)
    
    norm_out = np.clip(norm_out, 0, 255).astype(np.uint8)

    return norm_out

"""
def color_alignment_wavelet(img_org, img_trg, cal_region, plotting=False):
    content = torch.from_numpy(img_org.copy()).float().permute(2,0,1).unsqueeze(0)/255.
    style = torch.from_numpy(img_trg.copy()).float().permute(2,0,1).unsqueeze(0)/255.
    recon = wavelet_reconstruction(content, style)
    recon = recon[0].permute(1,2,0).numpy()*255.
    return recon 
"""

def color_alignment_wavelet(img_org, img_trg, cal_region, levels=5, plotting=False, return_torch=False):
    content, style = img_org, img_trg
    # import pdb; pdb.set_trace()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    recon = wavelet_reconstruction(content.to(device), style.to(device), levels)
    if return_torch:
        return recon
        return torch.clamp(recon, 0., 1) 
    recon = recon[0].cpu().permute(1,2,0).numpy()*255.
    recon = np.clip(recon, 0., 255.)
    return recon.astype(np.uint8)


def color_alignment_adain(img_org, img_trg, cal_region, plotting=False):
    src = img_org.astype(np.float32)
    des = img_trg.astype(np.float32)

    mean_src = np.mean(src, axis=(0,1))
    std_src = np.std(src, axis=(0,1))
    mean_des = np.mean(des, axis=(0,1))
    std_des = np.std(des, axis=(0,1))

    rtn = (img_org - mean_src) / std_src * std_des + mean_des
    return np.clip(rtn, 0, 255).astype(np.uint8)


def array2tensor(array, norm_type="0_1"):
    array = array.astype(np.float32) / 255.
    if norm_type == "-1_1":
        array = array * 2. - 1.
    tensor = torch.from_numpy(array).permute(2,0,1).unsqueeze(0)
    return tensor


def reblend(hq, lq, mask):
    hq = hq.astype(np.float32)
    lq = lq.astype(np.float32)
    mask = mask.astype(np.float32)

    return color_alignment_wavelet(array2tensor(hq), array2tensor(lq), None)
    
    # mask = mask/255.

    # rgba_hq = np.concatenate([hq, mask[...,np.newaxis]], axis=-1)
    # rgba_lq = np.concatenate([lq, mask[...,np.newaxis]], axis=-1)

    # Image.fromarray(rgba_hq.astype(np.uint8)).convert("RGBA").save("sample-hq.png")
    # Image.fromarray(rgba_lq.astype(np.uint8)).convert("RGBA").save("sample-lq.png")
    


def center_crop(arr):
    h, w = arr.shape[:2]
    if h >= w:
        d = h-w 
        return arr[d//2:d//2+w]
    else:
        d = w-h 
        return arr[:, d//2:d//2+h]


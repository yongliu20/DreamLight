import os
import cv2
import torch
from PIL import Image
import numpy as np 
from safetensors.torch import load_file
import sys 
try:
    from .fixdetails_utils.asym_vae import AsymmetricAutoencoderKL
except:
    from fixdetails_utils.asym_vae import AsymmetricAutoencoderKL

is_ascend = os.getenv('ASCEND_ENV', None) is not None
if is_ascend == False:
    try:
        from .fixdetails_utils.wavelet import wavelet_decomposition
    except:
        from fixdetails_utils.wavelet import wavelet_decomposition
else:
    from bytenn_ops import wavelet_decomposition
from diffusers import AutoencoderKL
from torchvision import transforms


class DetailsFixer(object):
    def __init__(self, model_path="vqmodel", model_size="large", shorter_size=512, max_longer_size=2048, device=torch.device("cuda:0")):
        if model_size == "large":
            self.model = AsymmetricAutoencoderKL.from_pretrained(model_path, extra_channels=3)
        elif model_size == "medium":
            self.model = AutoencoderKL.from_pretrained(model_path)
        self.model_size = model_size

        self.shorter_size = shorter_size
        self.max_longer_size = max_longer_size

        self.t_toTensor = transforms.ToTensor()
        self.t_norm = transforms.Normalize([.5,.5,.5], [.5,.5,.5])
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def resize_longer(self, image):
        h, w = image.shape[:2]
        if max(h, w) > self.max_longer_size:
            ratio = self.max_longer_size / max(h, w)
            image = cv2.resize(image, (int(w*ratio), int(h*ratio)))

    def blend(self, fg, canvas, fg_mask):
        fg_mask = cv2.erode(fg_mask, np.ones((3,3)), iterations=3)
        fg_mask = cv2.blur(fg_mask, (5, 5))
        fg_mask = fg_mask.astype(np.float32)/255.
        if len(fg_mask.shape) == 2:
            fg_mask = fg_mask[..., np.newaxis]
        result = canvas.astype(np.float32)*(1.-fg_mask) + fg.astype(np.float32)*fg_mask
        return np.clip(result, 0., 255.).astype(np.uint8)

    @torch.no_grad()
    def run(self, image1, image2, mask, ablation=False):
        image2_bp = image2.copy()
        mask_bp = mask.copy()
        if len(mask.shape) == 2:
            mask = np.stack([mask]*3, axis=-1)
        ys, xs = np.where(mask[...,0]>0)
        if xs.shape[0] < 1:
            return image2

        top, btm, lft, rit = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = mask.shape[:2]
        pads = [
            top, h-btm-1, lft, w-rit-1
        ]
        image1, image2, mask = image1[top:btm+1, lft:rit+1], image2[top:btm+1, lft:rit+1], mask[top:btm+1, lft:rit+1]
        patch_h, patch_w = image1.shape[:2]
        patch_size = min(patch_h, patch_w)
        ratio = self.shorter_size / patch_size
        patch_nh, patch_nw = int(patch_h*ratio), int(patch_w*ratio)
        patch_nh = max(self.shorter_size, patch_nh)
        patch_nw = max(self.shorter_size, patch_nw)
        patch_nh = patch_nh//8*8
        patch_nw = patch_nw//8*8
        image1 = Image.fromarray(image1).resize((patch_nw, patch_nh))
        image2 = Image.fromarray(image2).resize((patch_nw, patch_nh))
        mask = Image.fromarray(mask).resize((patch_nw, patch_nh))
        image1 = self.t_toTensor(image1)
        image2 = self.t_toTensor(image2)
        mask = self.t_toTensor(mask)
        image1 = self.t_norm(image1 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        image2 = self.t_norm(image2 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            hq1, lq1 = wavelet_decomposition(image1)
            hq2, lq2 = wavelet_decomposition(image2)
            mix12 = hq1+lq2 

        mask = torch.zeros_like(image2[:,:1]).to(mix12)
        if self.model_size == "large":
            alpha, beta = self.model(sample=hq1, mask=mask, condition=torch.cat([lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        elif self.model_size == "medium":
            alpha, beta = self.model(torch.cat([hq1, lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        hq1 = hq1 * alpha + beta

        result = hq1+lq2
        result = torch.nn.functional.interpolate(result, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        result = torch.nn.functional.pad(result, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)

        if ablation:
            mix12 = torch.nn.functional.interpolate(mix12, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
            mix12 = torch.nn.functional.pad(mix12, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)
            result = torch.cat([mix12, result], dim=-1)

        result = torch.clamp(result, -1., 1.)
        result = result[0] * .5 + .5
        result = (result*255.).permute(1,2,0).cpu().numpy().astype(np.uint8)

        # blend
        if ablation:
            h, w2 = result.shape[:2]
            results = [result[:,:w2//2], result[:,w2//2:]]
        else:
            results = [result]
        
        results = [self.blend(result, image2_bp, mask_bp) for result in results]


        return np.concatenate(results, axis=1)

    @torch.no_grad()
    def run_hr(self, image1, image2, mask, ablation=False):
        image2_bp = image2.copy()
        mask_bp = mask.copy()
        if len(mask.shape) == 2:
            mask = np.stack([mask]*3, axis=-1)
        ys, xs = np.where(mask[...,0]>0)
        if xs.shape[0] < 1:
            return image2

        top, btm, lft, rit = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = mask.shape[:2]
        pads = [
            top, h-btm-1, lft, w-rit-1
        ]
        image1, image2, mask = image1[top:btm+1, lft:rit+1], image2[top:btm+1, lft:rit+1], mask[top:btm+1, lft:rit+1]
        patch_h, patch_w = image1.shape[:2]
        patch_size = min(patch_h, patch_w)
        ratio = 1#self.shorter_size / patch_size
        patch_nh, patch_nw = int(patch_h*ratio), int(patch_w*ratio)
        patch_nh = max(self.shorter_size, patch_nh)
        patch_nw = max(self.shorter_size, patch_nw)
        patch_nh = patch_nh//8*8
        patch_nw = patch_nw//8*8
        image1 = Image.fromarray(image1).resize((patch_nw, patch_nh))
        image2 = Image.fromarray(image2).resize((patch_nw, patch_nh))
        mask = Image.fromarray(mask).resize((patch_nw, patch_nh))
        image1 = self.t_toTensor(image1)
        image2 = self.t_toTensor(image2)
        mask = self.t_toTensor(mask)
        image1 = self.t_norm(image1 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        image2 = self.t_norm(image2 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            hq1, lq1 = wavelet_decomposition(image1)
            hq2, lq2 = wavelet_decomposition(image2)
            mix12 = hq1+lq2 

        mask = torch.zeros_like(image2[:,:1]).to(mix12)
        if self.model_size == "large":
            alpha, beta = self.model(sample=hq1, mask=mask, condition=torch.cat([lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        elif self.model_size == "medium":
            alpha, beta = self.model(torch.cat([hq1, lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        hq1 = hq1 * alpha + beta

        result = hq1+lq2
        result = torch.nn.functional.interpolate(result, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        result = torch.nn.functional.pad(result, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)

        if ablation:
            mix12 = torch.nn.functional.interpolate(mix12, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
            mix12 = torch.nn.functional.pad(mix12, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)
            result = torch.cat([mix12, result], dim=-1)

        result = torch.clamp(result, -1., 1.)
        result = result[0] * .5 + .5
        result = (result*255.).permute(1,2,0).cpu().numpy().astype(np.uint8)

        # blend
        if ablation:
            h, w2 = result.shape[:2]
            results = [result[:,:w2//2], result[:,w2//2:]]
        else:
            results = [result]
        
        results = [self.blend(result, image2_bp, mask_bp) for result in results]


        return np.concatenate(results, axis=1)

    @torch.no_grad()
    def run_mixres(self, image1, image2, mask, ablation=False):
        image2_bp = image2.copy()
        mask_bp = mask.copy()
        if len(mask.shape) == 2:
            mask = np.stack([mask]*3, axis=-1)
        ys, xs = np.where(mask[...,0]>0)
        if xs.shape[0] < 1:
            return image2

        top, btm, lft, rit = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = mask.shape[:2]
        pads = [
            top, h-btm-1, lft, w-rit-1
        ]
        image1, image2, mask = image1[top:btm+1, lft:rit+1], image2[top:btm+1, lft:rit+1], mask[top:btm+1, lft:rit+1]
        patch_h, patch_w = image1.shape[:2]
        patch_size = min(patch_h, patch_w)
        ratio = self.shorter_size / patch_size
        patch_nh, patch_nw = int(patch_h*ratio), int(patch_w*ratio)
        patch_nh = max(self.shorter_size, patch_nh)
        patch_nw = max(self.shorter_size, patch_nw)
        patch_nh = patch_nh//8*8
        patch_nw = patch_nw//8*8
        # hr
        image1_hr = image1
        image2_hr = image2
        mask_hr = mask
        image1_hr = self.t_toTensor(image1_hr)
        image2_hr = self.t_toTensor(image2_hr)
        mask_hr = self.t_toTensor(mask_hr)
        image1_hr = self.t_norm(image1_hr * mask_hr + (1.-mask_hr) * 0.5).unsqueeze(0).to(self.device)
        image2_hr = self.t_norm(image2_hr * mask_hr + (1.-mask_hr) * 0.5).unsqueeze(0).to(self.device)
        # lr
        image1 = Image.fromarray(image1).resize((patch_nw, patch_nh))
        image2 = Image.fromarray(image2).resize((patch_nw, patch_nh))
        mask = Image.fromarray(mask).resize((patch_nw, patch_nh))
        image1 = self.t_toTensor(image1)
        image2 = self.t_toTensor(image2)
        mask = self.t_toTensor(mask)
        image1 = self.t_norm(image1 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        image2 = self.t_norm(image2 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            hq1, lq1 = wavelet_decomposition(image1)
            hq2, lq2 = wavelet_decomposition(image2)
            mix12 = hq1+lq2 

        mask = torch.zeros_like(image2[:,:1]).to(mix12)
        if self.model_size == "large":
            alpha, beta = self.model(sample=hq1, mask=mask, condition=torch.cat([lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        elif self.model_size == "medium":
            alpha, beta = self.model(torch.cat([hq1, lq2], dim=1), return_dict=False, sample_posterior=False)[0].chunk(2, dim=1)
        
        alpha = torch.nn.functional.interpolate(alpha, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        beta = torch.nn.functional.interpolate(beta, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        hq1, _ = wavelet_decomposition(image1_hr, levels=3)
        _, lq2 = wavelet_decomposition(image2_hr, levels=3)
        hq1 = hq1 * alpha + beta
        result = hq1+lq2
        result = torch.nn.functional.pad(result, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)

        
        if ablation:
            mix12 = torch.nn.functional.interpolate(mix12, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
            mix12 = torch.nn.functional.pad(mix12, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)
            result = torch.cat([mix12, result], dim=-1)

        result = torch.clamp(result, -1., 1.)
        result = result[0] * .5 + .5
        result = (result*255.).permute(1,2,0).cpu().numpy().astype(np.uint8)

        # blend
        if ablation:
            h, w2 = result.shape[:2]
            results = [result[:,:w2//2], result[:,w2//2:]]
        else:
            results = [result]
        
        results = [self.blend(result, image2_bp, mask_bp) for result in results]
        return np.concatenate(results, axis=1)
    

    @torch.no_grad()
    def run_wavelet(self, image1, image2, mask, ablation=False):
        image2_bp = image2.copy()
        mask_bp = mask.copy()
        if len(mask.shape) == 2:
            mask = np.stack([mask]*3, axis=-1)
        ys, xs = np.where(mask[...,0]>0)
        if xs.shape[0] < 1:
            return image2

        top, btm, lft, rit = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = mask.shape[:2]
        pads = [
            top, h-btm-1, lft, w-rit-1
        ]
        image1, image2, mask = image1[top:btm+1, lft:rit+1], image2[top:btm+1, lft:rit+1], mask[top:btm+1, lft:rit+1]
        patch_h, patch_w = image1.shape[:2]
        patch_size = min(patch_h, patch_w)
        ratio = self.shorter_size / patch_size
        patch_nh, patch_nw = int(patch_h*ratio), int(patch_w*ratio)
        patch_nh = max(self.shorter_size, patch_nh)
        patch_nw = max(self.shorter_size, patch_nw)
        patch_nh = patch_nh//8*8
        patch_nw = patch_nw//8*8
        # hr
        image1_hr = image1
        image2_hr = image2
        mask_hr = mask
        image1_hr = self.t_toTensor(image1_hr)
        image2_hr = self.t_toTensor(image2_hr)
        mask_hr = self.t_toTensor(mask_hr)
        image1_hr = self.t_norm(image1_hr * mask_hr + (1.-mask_hr) * 0.5).unsqueeze(0).to(self.device)
        image2_hr = self.t_norm(image2_hr * mask_hr + (1.-mask_hr) * 0.5).unsqueeze(0).to(self.device)
        # lr
        image1 = Image.fromarray(image1).resize((patch_nw, patch_nh))
        image2 = Image.fromarray(image2).resize((patch_nw, patch_nh))
        mask = Image.fromarray(mask).resize((patch_nw, patch_nh))
        image1 = self.t_toTensor(image1)
        image2 = self.t_toTensor(image2)
        mask = self.t_toTensor(mask)
        image1 = self.t_norm(image1 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        image2 = self.t_norm(image2 * mask + (1.-mask) * 0.5).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            hq1, lq1 = wavelet_decomposition(image1)
            hq2, lq2 = wavelet_decomposition(image2)
            mix12 = hq1+lq2 
        result = mix12
        result = torch.nn.functional.interpolate(result, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        result = torch.nn.functional.pad(result, (pads[2], pads[3], pads[0], pads[1]), mode="constant", value=0)

        result = torch.clamp(result, -1., 1.)
        result = result[0] * .5 + .5
        result = (result*255.).permute(1,2,0).cpu().numpy().astype(np.uint8)

        # blend
        if ablation:
            h, w2 = result.shape[:2]
            results = [result[:,:w2//2], result[:,w2//2:]]
        else:
            results = [result]
        
        results = [self.blend(result, image2_bp, mask_bp) for result in results]
        return np.concatenate(results, axis=1)


    
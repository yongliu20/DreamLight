import os
import math
import numpy as np
import torch
import safetensors.torch as sf
import db_examples
from torchvision import transforms

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import cv2
from ip_adapter import IPAdapterPlus
from tqdm import tqdm
import sys
sys.path.append("SD15/FixDetails")
from detail_fixer import DetailsFixer
detail_fixer = DetailsFixer(model_path="FixDetails/vqmodel")


# adapter
image_encoder_path = "ckpt/CLIP/models"
ip_ckpt = 'ckpt/SD15/ip_ckpt/adapter.bin'
device = "cuda"

sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet", revision=None)
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")


# Change UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in
unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load
model_path = "ckpt/SD15/model.safetensors"

# if not os.path.exists(model_path):
#     download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] * 0 + sd_offset['unet.' + k] for k in sd_origin.keys()}
# sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=ddim_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

ip_model = IPAdapterPlus(t2i_pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    token_ids = torch.tensor(tokens).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids)[0]

    return conds

@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results

@torch.inference_mode()
def pytorch2numpy_new(imgs, quant=True):
    if quant:
        results = []
        for img in imgs:
            img = img * 0.5 + 0.5     # Unnormalize
            img = img.movedim(0, -1) * 255
            img = img.clamp(0, 255)     # Clamp values to be between 0 and 1
            img = img.detach().float().cpu().numpy().astype(np.uint8)         # Convert back to numpy array
            results.append(img)
    else:
        results = []
        for img in imgs:
            img = img * 0.5 + 0.5     # Unnormalize
            img = img.movedim(0, -1)
            img = img.clamp(0, 1)     # Clamp values to be between 0 and 1
            img = img.detach().float().cpu().numpy().astype(np.float32)         # Convert back to numpy array
            results.append(img)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

@torch.inference_mode()
def numpy2pytorch_new(imgs):
    # h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    # h = h.movedim(-1, 1)
    transform1 = transforms.ToTensor()
    transform2 = transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    hs = []
    for img in imgs:
        h = transform1(img)
        h = transform2(h)
        hs.append(h)
    h = torch.stack(hs, dim=0)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def resize_and_center_crop_mask(image, target_width, target_height):
    image = np.squeeze(image) * 255
    image = image.astype(np.uint8)
    pil_image = Image.fromarray(image).convert('RGB')
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

@torch.inference_mode()
def run_rmbg_gt(img, mask):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, matting, use_fix=True):
    # bg_source: text or image

    rng = torch.Generator(device=device).manual_seed(seed)

    # image_width, image_height = 512, 512
    # TODO
    h0, w0 = input_fg.shape[:2]
    ratio = 512/max(w0, h0)
    w, h = int(ratio*w0), int(ratio*h0)
    w = w//16*16
    h = h//16*16
    image_height, image_width = h, w
    image_height, image_width = 512, 512
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    if bg_source == 'image':
        bg = resize_and_center_crop(input_bg, image_width, image_height)
    else:
        bg = np.zeros_like(fg)
    # mask
    matting = resize_and_center_crop_mask(matting, image_width, image_height)
    # cv2.imwrite('/mask.jpg', matting)
    matting_array = np.zeros(matting.shape[:2])
    matting_array[matting[:, :, 0] > 128] = 1
    matting_array = np.uint8(matting_array * 255)

    # mask
    matting = numpy2pytorch_new([matting/255]).to(device=vae.device, dtype=vae.dtype)
    matting = matting * 0.5 + 0.5   # (b, c, h, w)
    matting = torch.mean(matting, dim=1, keepdim=True)    # (b, 1, H, W)
    # matting = (matting > 0.5).float()
    # print(matting.sum())

    fg_array = fg.copy()
    
    
    concat_conds = numpy2pytorch_new([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    
    concat_conds[0] = matting * concat_conds[0]
    concat_conds[1] = (1 - matting) * concat_conds[1]
    if bg_source == 'text':
        concat_conds[1] = torch.zeros_like(matting) * concat_conds[1]
        bg = np.ones_like(bg) * 127

    # adapter
    pil_bg = Image.fromarray(bg).convert("RGB")


    fg, bg = pytorch2numpy_new(concat_conds)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    # concat_conds = vae.encode(concat_conds).latent_dist.mode()
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    
    latents = ip_model.generate_new(
        pil_image=pil_bg, 
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
        matting=matting,
        ).to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy_new(pixels, quant=False)
    for i, p in enumerate(pixels):
        if use_fix:
            fixed_p = detail_fixer.run_mixres(fg_array, np.uint8(p*255), matting_array)
            pixels[i] = fixed_p
        else:
            pixels[i] = np.uint8(p*255)
        # pixels[i] = np.uint8(p*255)
    return pixels, [fg, bg]

@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, gt_mask_path):
    # input_fg, matting = run_rmbg(input_fg)
    # print(matting.shape, np.unique(matting))
    # exit()

    # use gt mask
    gt_mask = cv2.imread(gt_mask_path, 0)
    # TODO
    # gt_mask[gt_mask >= 128] = 255
    # gt_mask[gt_mask < 128] = 0
    gt_mask = np.expand_dims(gt_mask, axis=-1) / 255
    # gt_mask[gt_mask > 0] = 1
    # gt_mask[gt_mask < 1] = 0
    matting = gt_mask
    # input_fg = (input_fg * matting).astype(np.uint8)

    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, matting)
    # results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images

def inference(input_fg_path, input_bg_path=None, prompt="", image_width=512, image_height=512, num_samples=1, seed=42, steps=25, a_prompt="best quality", n_prompt="low resolution, bad anatomy, bad hands, cropped, worst quality", cfg=3.5, highres_scale=1.5, highres_denoise=0.5, save_root=None, gt_mask_path=None):
    input_fg = cv2.imread(input_fg_path)
    input_fg = cv2.cvtColor(input_fg, cv2.COLOR_BGR2RGB)
    if input_bg_path:
        input_bg = cv2.imread(input_bg_path)
        input_bg = cv2.cvtColor(input_bg, cv2.COLOR_BGR2RGB)
        bg_source="image"
    else:
        input_bg = None
        bg_source="text"
    # all_outputs[0]: relit, all_outputs[1]: fg, all_outputs[2]: bg
    all_outputs = process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, gt_mask_path=gt_mask_path)
    relit_result = all_outputs[0]
    relit_result = cv2.cvtColor(relit_result, cv2.COLOR_RGB2BGR)
    fg = all_outputs[1]
    fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
    bg = all_outputs[2]
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    if save_root:
        if input_bg_path:
            save_path = os.path.join(save_root, os.path.basename(input_bg_path).split('.png')[0], os.path.basename(input_fg_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = os.path.join(save_root, os.path.basename(input_fg_path))
        # cv2.imwrite(save_path, relit_result)
        # cv2.imwrite(save_path.replace('.png', '_fg.png'), fg)
        # cv2.imwrite(save_path.replace('.png', '_bg.png'), bg)

        # TODO:
        if gt_mask_path is not None:
            # use gt mask
            gt_mask = cv2.imread(gt_mask_path, 0)
            gt_mask = np.expand_dims(gt_mask, axis=-1) / 255
            # TODO
            matting = resize_and_center_crop_mask(gt_mask, 512, 512)
            matting_array = np.zeros(matting.shape[:2])
            matting_array[matting[:, :, 0] > 128] = 1
            matting_array = np.uint8(matting_array * 255)


    return relit_result




if __name__ == "__main__":
    val_dataset_root = "xxx/xxx"
    fg_root = os.path.join(val_dataset_root, "fgs")
    bg_root = os.path.join(val_dataset_root, "bgs")
    mask_root = os.path.join(val_dataset_root, "masks")
    prompt_root = os.path.join(val_dataset_root, "prompts-ly")

    initial_save_bg_root = 'xxx/xxx'
    initial_save_text_root = 'xxx/xxx'
    os.makedirs(initial_save_bg_root, exist_ok=True)
    os.makedirs(initial_save_text_root, exist_ok=True)

    bg_paths = ['background.png']

    for i in tqdm(os.listdir(fg_root)):
        input_fg_path = os.path.join(fg_root, i)
        gt_mask_path = os.path.join(mask_root, i)

        for bbb in bg_paths:
            input_bg_path = bbb
            fn = i[:-1-len(i.split(".")[-1])]
            prompt_path = os.path.join(prompt_root, fn+".txt")
            if os.path.exists(prompt_path):
                fh = open(prompt_path, "r", encoding="utf-8")
                prompt = ""
                lines = fh.readlines()
                fh.close()
                prompt = " ".join([line.strip() for line in lines])
            else:
                prompt = "harmonious, natural, photorealistic, seamless, homogeneous"

            try:
                result = inference(input_fg_path=input_fg_path, input_bg_path=input_bg_path, prompt='harmonious, natural, photorealistic, seamless, homogeneous', save_root=initial_save_bg_root, gt_mask_path=gt_mask_path)
                result = inference(input_fg_path=input_fg_path, prompt=prompt, save_root=initial_save_text_root, gt_mask_path=gt_mask_path)
                break
            except:
                print(input_bg_path)
        # 

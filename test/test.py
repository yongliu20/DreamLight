import os
import shutil, cv2
import torch
from tqdm import tqdm 
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxPipeline,
    FluxTransformer2DModel,
)
from dreamlight_utils.pipeline_flux import FluxPipeline
from PIL import Image
import numpy as np
from dreamlight_utils.DreamlightDataFactory.utils.sh_convert import read_envmap, spherical_harmonics_coeffs_v2, generate_spherical_image_v2
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F


CKPT_ROOT = "xxx/xxx"
DATA_ROOT = "xxx/xxx"


def inverse_transform(tensor):
    tensor = tensor * 0.5 + 0.5
    array = tensor.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)


class Args:
    def __init__(self):
        self.pretrained_model_name_or_path = "xxx/xxx" # Pretrained FLUX.1-dev path
        self.revision = None 
        self.variant = None
        self.weight_dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        self.seed = 19971021
        self.output_model_pth = "ckpt/transformer/model.pth"
        self.clipvision_model_path = 'ckpt/CLIP/models'
        
        self.add_bg = True
        self.add_env = True

if __name__ == "__main__":
    args = Args()
    weight_dtype = args.weight_dtype

    

    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        device_map=None, 
        low_cpu_mem_usage=False,
    )

    transformer = pipeline.transformer
    extra_channels = int(args.add_bg) + int(args.add_env)
    with torch.no_grad():
        x_embedder = transformer.x_embedder
        new_x_embedder = torch.nn.Linear(x_embedder.in_features*(1+1+extra_channels), x_embedder.out_features)
        new_x_embedder.weight.data.zero_()
        new_x_embedder.weight.data[:, :x_embedder.in_features].copy_(x_embedder.weight.data)
        new_x_embedder.bias.data.copy_(x_embedder.bias.data)
        transformer.x_embedder = new_x_embedder.to(weight_dtype)
    transformer.load_state_dict(torch.load(args.output_model_pth, map_location="cpu"))

    pipeline.to(args.device)

    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.clipvision_model_path).to(device=pipeline.device)

    #### main
    val_dataset_root = "xxx/xxx"    # test_data_root 
    fg_root = os.path.join(val_dataset_root, "fgs")
    bg_root = os.path.join(val_dataset_root, "bgs")
    mask_root = os.path.join(val_dataset_root, "masks")
    prompt_root = os.path.join(val_dataset_root, "prompts")

    initial_save_root = 'xxx/xxx'
    os.makedirs(initial_save_root, exist_ok=True)

    sample = 'xxx/xxx'
    bg_paths = ["xxx/xxx"]
    for bbb in bg_paths:
        bg_path = bbb
        env_path = bbb
        save_root = os.path.join(initial_save_root, os.path.basename(bg_path).split('.png')[0])
        if os.path.exists(save_root) == False:
            os.mkdir(save_root)
        print("Processing", sample, "...")
        fg_path = os.path.join(fg_root, sample)
        mask_path = os.path.join(mask_root, sample)
        fn = sample[:-1-len(sample.split(".")[-1])]
        prompt_path = os.path.join(prompt_root, fn+".txt")
        if os.path.exists(prompt_path):
            fh = open(prompt_path, "r", encoding="utf-8")
            prompt = ""
            lines = fh.readlines()
            fh.close()
            prompt = " ".join([line.strip() for line in lines])

        else:
            prompt = "harmonious, natural, photorealistic, seamless, homogeneous"

        fg = Image.open(fg_path).convert("RGB")
        bg = Image.open(bg_path).convert("RGB")
        env= Image.open(env_path).convert("RGB")
        w0, h0 = fg.size 
        ratio = 1024/max(w0, h0)
        w, h = int(ratio*w0), int(ratio*h0)
        w = w//16*16
        h = h//16*16
        fg = np.array(fg.resize((w, h)))
        bg = np.array(bg.resize((w, h)))
        env = np.array(env.resize((w, h)))
        mask = np.array(Image.open(mask_path).convert("RGB").resize((w, h)))
        matting = mask.astype(np.float32)/255.
        
        # fg
        fg = fg.astype(np.float32)*matting + 127.*(1.-matting)
        fg = np.clip(fg, 0., 255.)
        fg = fg/255.*2.-1.
        cond_fg_values = torch.from_numpy(fg).permute(2,0,1).unsqueeze(0).to(pipeline.device)
        # bg
        bg = bg.astype(np.float32)*(1.-matting) + 127.*(matting)
        bg = np.clip(bg, 0., 255.)
        bg = bg*1./255.*2.-1.
        cond_bg_values = torch.from_numpy(bg).permute(2,0,1).unsqueeze(0).to(pipeline.device)
        # env
        ldr_image = env.astype(np.float32)/255.
        gamma = 2.2
        self_envmap = np.power(ldr_image, 1/gamma)
        size = 256
        self_envmap_crop = self_envmap
        coeffs = spherical_harmonics_coeffs_v2(self_envmap*255, order=2, num_samples=10000)
        env_lighting = generate_spherical_image_v2(coeffs, self_envmap.shape[1], self_envmap.shape[0]) # TODO
        env_lighting = torch.from_numpy(env_lighting).float().permute(2,0,1)/127.5-1
        cond_env_values = env_lighting.unsqueeze(0).to(pipeline.device)
        # seed
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed) if args.seed else None

        pipeline_args = {
            "prompt":prompt, "height":h, "width":w,
        }
        
        results = []
        for (w_fg, w_bg, w_env) in [(1, 0, 0), (0, 1, 0)]:
            if w_bg == 1:
                pipeline_args["prompt"] = "harmonious, natural, photorealistic, seamless, homogeneous"
            dreamlight_kwargs = {
                "cond_fg_values":cond_fg_values*w_fg, 
                "cond_bg_values":cond_bg_values*w_bg, 
                "cond_env_values":cond_env_values*w_env, 
            }
            
            clip_bg_img = inverse_transform(cond_bg_values.squeeze(0)*w_bg)
            clip_bg_img = clip_image_processor(images=clip_bg_img, return_tensors="pt").pixel_values
            clip_bg_img = clip_bg_img.to(device=pipeline.device)    # (b, c, h, w)
            clip_image_embeds = image_encoder(clip_bg_img, output_hidden_states=True).hidden_states[-2] #[b, l, c]

            fg_region_mask = torch.from_numpy(matting).permute(2,0,1).unsqueeze(0).to(pipeline.device)
            fg_region_mask = F.interpolate(fg_region_mask, scale_factor=1/16,  mode='nearest')
            mask_h, mask_w = fg_region_mask.shape[-2:]
            fg_region_mask = fg_region_mask.flatten(2).transpose(1, 2)[:,:,0:1] # [B, L, 1]

            dreamlight_kwargs.update({"ip_adapter_masks": fg_region_mask, 'ip_adapter_image_embeds': clip_image_embeds})

            with torch.no_grad():
                image = pipeline(
                    **pipeline_args, 
                    generator=generator, 
                    dreamlight_kwargs=dreamlight_kwargs,
                    joint_attention_kwargs={"ip_adapter_h": mask_h, "ip_adapter_w": mask_w}
                    ).images[0]
            results.append(np.array(image))
        image = Image.fromarray(np.concatenate(results, axis=1))
        image.save(os.path.join(save_root, "{}.png".format(prompt)))
    

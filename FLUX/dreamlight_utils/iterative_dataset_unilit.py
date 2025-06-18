import bson
import math
import random
import json
import os
import torch
import sys
from dataclasses import dataclass, field
from dataloader import KVReader
from PIL import Image, ImageDraw
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from typing import Any, Dict, List, Callable, Optional, Tuple
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time 

from dreamlight_utils.dataset_utils.dataset_common import ImageDecoder, MaskDecoder, TextDecoder, MultiSourceTextDecoder,MultiSourceTextDecoder_aesFinetune, MultiSourceTextDecoder_inpaint
from dreamlight_utils.dataset_utils.timer import timer
from dreamlight_utils.dataset_utils.hdfs import listdir
from dreamlight_utils.dataset_utils.utils import partition_by_size
from dreamlight_utils.dataset_utils.prompt2image import prompt2image

from dreamlight_utils.DreamlightDataFactory.image_degradation import IBRImageDegradation
from dreamlight_utils.DreamlightDataFactory.utils.sh_convert import read_envmap, spherical_harmonics_coeffs_v2, generate_spherical_image_v2


@dataclass
class Bucket:
    index_files: List[str] = field(default_factory=list) # the .index filenames
    image_count: int       = field(default=0)            # the total number of images
    image_height: int      = field(default=0)            # the image height
    image_width: int       = field(default=0)            # the image width


def gen_lighting_mask(grid_size=(4,4), target_size=(32,32)):
    mask = np.random.uniform(0, 1, grid_size)
    # mask = np.array(
    #     [[1,1,1,0], [1,1,1,0], [0,1,0,1], [0,1,0,1]]
    # )
    # mask = (mask>0.5).astype(np.uint8)
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, target_size, cv2.INTER_LINEAR)
    return mask




def check_image(image_path):
    iftype = image_path.split(".")[-1]
    path = image_path[:-1-len(image_path.split(".")[-1])]
    for postfix in [iftype, "jpg", "png", "bmp", "jpeg", "PNG", "webp", "JPEG", "BMP"]:
        if os.path.exists(path+"."+postfix):
            return path+"."+postfix
    return image_path


class DreamlightImageMultiscaleDataset(IterableDataset):
    def __init__(self, image_root, mask_root, envmap_root, meta_root, eigen_root, hard_shadow_root,
        batch_size:int=1,
        rank:int=0,
        world_size:int=1,
        bucket_sharding=False,
        bucket_override:Optional[List[Tuple[int, int]]]=None,
        resolution:int=512,
        debug=False,
        bucket_ratio=1.,
        for_fusion=False,
        for_bginpainting=False,
        enable_timer=False,
    ):
        self.image_root = image_root
        self.mask_root = mask_root
        self.envmap_root = envmap_root
        self.meta_root = meta_root
        if isinstance(eigen_root, (tuple, list)):
            self.eigen_root = eigen_root
        else:
            self.eigen_root = [eigen_root]
        self.hard_shadow_root = hard_shadow_root
        self.batch_size = batch_size
        self.rank = rank
        self.for_fusion = for_fusion
        self.for_bginpainting = for_bginpainting
        self.enable_timer = enable_timer

        bucket_override_rescaled = []
        for h, w in bucket_override:
            h = int(h*bucket_ratio)
            w = int(w*bucket_ratio)
            bucket_override_rescaled.append((h, w))
        self.bucket_override = bucket_override_rescaled
        self.resolution = resolution
        self.debug = debug
        self.samples = {}

        paths = listdir(meta_root)
        if debug:
            paths = [path for path in paths if os.path.basename(path) in debug]
        # paths = [path for path in paths if os.path.basename(path) not in ["v5", "v6"]]

        self.buckets = self.get_buckets(sum([listdir(path) for path in paths], []))
        self.bucket_entries = list(self.buckets.values())
        self.bucket_weights = list(map(lambda bucket:bucket.image_count, self.bucket_entries))
        self.bucket_iterators = list(map(lambda bucket:self._iterate_bucket(bucket), self.bucket_entries))
        # breakpoint()
        self.length = 0
        for k in self.buckets:
            self.length += self.buckets[k].image_count
        # self.sample_weights = []
        # self.pointers = {}
        # for k in self.samples:
        #     l = len(self.samples[k])
        #     self.sample_weights.append(l)
        #     self.length += l
        #     self.pointers[k] = 0
        # self.bucket_sizes = list(self.samples.keys())

        if bucket_sharding:
            assert len(self.buckets.keys()) < world_size, (
                f"world_size: {world_size} must be greater than the amount of " + 
                f"buckets: {len(self.buckets.keys())} for sharding to work."
            )
            bucket_entries = list(self.buckets.values())
            if rank < len(bucket_entries):
                # When the current GPU rank is less than total buckets, directly assign it to the bucket at that index.
                # This ensures each bucket is assigned to at least one GPU.
                bucket_idx_for_rank = rank
            else:
                # For additional GPUs, choice a bucket based on weights.
                # First, compute each bucket weights.
                bucket_image_counts = torch.tensor(list(map(lambda bucket:bucket.image_count, bucket_entries)))
                bucket_weights = bucket_image_counts / bucket_image_counts.sum()
                # Then, compute how many GPUs should be assigned for each bucket.
                num_remaining_gpus = world_size - len(bucket_entries)
                num_gpus_assigned_per_bucket = (bucket_weights * num_remaining_gpus).round()
                # Finally, find out which bucket should be assigned to the current rank.
                bucket_idx_for_rank = num_gpus_assigned_per_bucket.cumsum(0).gt(rank).int().argmax()
            
            # Rewrite the bucket dictionary to only keep the selected bucket.
            bucket_for_rank = bucket_entries[bucket_idx_for_rank]
            bucket_for_rank_key = (bucket_for_rank.image_width, bucket_for_rank.image_height)
            self.buckets = {bucket_for_rank_key: bucket_for_rank}

            if self.debug:
                print(
                    f"Selected bucket index: {bucket_idx_for_rank} " +
                    f"of size ({bucket_for_rank.image_width}x{bucket_for_rank.image_height}) " +
                    f"for rank: {rank}, world_size: {world_size}."
                )
        self.t_toTensor = transforms.ToTensor()
        self.t_norm = transforms.Normalize((.5,.5,.5), (.5,.5,.5))
        
        envmap_root_exr = [os.path.join(self.envmap_root, dataset, "hdr") for dataset in os.listdir(self.envmap_root)]
        envmap_root_hdri = ["/mnt/bn/xiaowenpeng-bytenas-lq4/projects/Harmonization/git-clones/Relighting/HDRIS"]
        envmap_root_image = ["/mnt/bn/dreamlight-bytenas-lq/dataset_render_duallights/lights"]
        self.image_degrader = IBRImageDegradation(
            envmap_root=envmap_root_exr+envmap_root_hdri+envmap_root_image,
            hardshadow_root=self.hard_shadow_root,
            envmap_shorter_size=256,
            overwrite_cache=False,
        )

    def __len__(self):
        return self.length

    def get_buckets(self, paths):
        buckets = {}
        for filepath in paths:
            # Parse name, example:
            #  filepath:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196.index"
            #  filename:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196"
            #  basename:  "2_19_256-896_00002_00196"
            #  extension: ".index"
            filename, extension = os.path.splitext(filepath)
            basename = os.path.basename(filename)

            # Parse basename, example:
            #  {id}_{image_count}_{image_height}-{image_width}_{other_info}
            if extension in [".index", ".json", ".snappy"] and "tempstate" not in filename:
                if self.debug:
                    print(f"dataset loading current filename: {filename}")
                image_count, image_height, image_width = basename.replace("_", "-").split("-")[1:4]
                image_count = int(image_count)
                image_height = int(image_height)
                image_width = int(image_width)

                # Override bucket resolution if configured.
                image_width, image_height = self._override_resolution_if_needed(image_width, image_height)

                bucket_key = (image_width, image_height)
                bucket_entry = buckets.get(bucket_key, Bucket())
                bucket_entry.index_files.append(filename+'.json')
                bucket_entry.image_count += image_count
                bucket_entry.image_height = image_height
                bucket_entry.image_width = image_width
                buckets[bucket_key] = bucket_entry
        counts = 0
        for i, bucket_entry in enumerate(buckets.values()):
            # print(
            #     f"Bucket {i}: {bucket_entry.image_width}x{bucket_entry.image_height} " +
            #     f"contains {bucket_entry.image_count} images."
            # )
            counts += bucket_entry.image_count
        print(f'=====> Paths contains {counts} images.')
        return buckets

    def _override_resolution_if_needed(self, width: int, height: int) -> Tuple[int, int]:
        """
        Override the bucket resolution if configured:
        Example:
            - bucket override: [(1000, 200), (200, 1000)]
            - current resolution: (300, 900)
            - return (200, 1000) because it is the closest in aspect ratio.
        """
        # If no bucket override, directly return original bucket width and height.
        if self.bucket_override is None:
            return width, height
        # If bucket override is defined, find a new resolution from the override list that best matches the aspect ratio.
        assert len(self.bucket_override) > 0, "bucket_override must not be an empty list."
        target_aspect_ratio = width / height
        bucket_resolutions = self.bucket_override
        bucket_aspect_ratios = torch.tensor([w / h for w, h in bucket_resolutions], dtype=torch.float64)
        bucket_idx = bucket_aspect_ratios.sub(target_aspect_ratio).abs().argmin().item()
        width, height = bucket_resolutions[bucket_idx]
        
        if self.resolution != 512:
            # The buckets are defined in 512 resolution. If target resolution is not 512, we need to scale it and make sure divisible by 64.
            ratio = self.resolution / 512
            width = (width * ratio) // 64 * 64
            height = (height * ratio) // 64 * 64

        return int(width), int(height)

    def _iterate_bucket(self, bucket: Bucket):
        # Copy the list, json_files
        # import pdb; pdb.set_trace()
        index_files = list(bucket.index_files)
        # kkey = (bucket.image_height, bucket.image_width)
        # if kkey not in self.samples:
        #     self.samples[kkey] = []

        while True:
            # Loop through all the .index files
            random.shuffle(index_files)
            for index_file in index_files:
                if True:#try:
                    with timer(
                            op=f"[Rank:{self.rank}] KVReader opens and lists keys from index file {index_file}",
                            wait_seconds=3
                    ):
                        with open(index_file, 'r') as f: #+'.json'
                            reader = json.load(f)
                            keys = list(reader.keys())
                    random.shuffle(keys)
                    for key_batch in partition_by_size(keys, self.batch_size * 8):
                        for i, data in enumerate(key_batch):
                            s_time = time.time()
                            probability = random.random()
                            meta_data = reader[data]

                            # degradation materials
                            envmap_path = os.path.join(self.envmap_root, data.split("/")[0], "hdr", data.split("/")[1]+".exr")
                            
                            for e_root in self.eigen_root:
                                albedo_path = os.path.join(e_root, data, "albedo.png")
                                normal_path = os.path.join(e_root, data, "normal.png")
                                if os.path.exists(albedo_path) and os.path.exists(normal_path):
                                    break
                            
                            tags = ["prompt", "bgprompt", "bg_prompt_gpt4o", "v3_0_no_title_internlm_caption_en_text"]
                            tags = [tag for tag in tags if tag in meta_data]
                            if len(tags) == 0:
                                tags = "prompt"
                            tag = random.choice(tags)
                            caption = meta_data.get(tag, "")

                            e_time = time.time()
                            image_path = check_image(os.path.join(self.image_root, data+".jpg"))
                            mask_path = check_image(os.path.join(self.mask_root, data+".png"))
                            sample = [bucket, caption, image_path, mask_path, envmap_path, albedo_path, normal_path]
                            
                            existance = True
                            # for path in [image_path, mask_path, envmap_path, albedo_path, normal_path]:
                            for path in [image_path, mask_path, albedo_path, normal_path]:
                                if os.path.exists(path) == False:
                                    print(">>>>", path)
                                    existance = False
                                    break
                            if existance:
                                # self.samples[kkey].append(sample)
                                yield sample
                        break

                    # print("++ Samples size at {}:".format(kkey), len(self.samples[kkey]))

                # except Exception as ex:
                #     # breakpoint()
                #     # Error may happen due to network issue when reading from data from this file.
                #     # Skip to the next index file regardless.
                #     print(f"Bucket dataset reading data received unexpected exception at file: {index_file}", ex)
                #     continue

    def make_batch(self, samples):
        results = {}
        sample = samples[0]
        for k in sample:
            # print(type(sample[k]))
            values = [sample[k] for sample in samples]
            if type(sample[k]) == torch.Tensor:
                results[k] = torch.stack(values, dim=0)
            elif type(sample[k]) == str:
                results[k] = values
            else:
                raise NotImplementedError
        return results

            
    def __iter__(self):
        while True:
            bucket_iterator = random.choices(self.bucket_iterators, self.bucket_weights)[0]
            sample_returned = 0
            samples = []
            timer = {
                "load_materials":[],
                "prepare_LML1L2":[],
                "degradation":[],
                "Aug&toTensor":[],
            }
            
            sh_num_steps = 100000
            while sample_returned < self.batch_size:
                s_time = time.time()
                bucket, text, image_path, mask_path, envmap_path, albedo_path, normal_path = next(bucket_iterator)

                s_time = time.time()
                image = Image.open(image_path).convert("RGB")

                W, H = image.size
                if image.size[0] < bucket.image_width or image.size[1] < bucket.image_height:
                    print("Fail here.")
                    # continue

                mask = Image.open(mask_path).convert("RGB")
                try:
                    normal = Image.open(normal_path).convert("RGB").resize((W,H))
                    albedo = Image.open(albedo_path).convert("RGBA").resize((W,H))
                    if "switchlight" in albedo_path:
                        albedo_alpha = np.array(albedo)
                        mask = Image.fromarray(albedo_alpha[...,3]).convert("RGB")
                        albedo = Image.fromarray(albedo_alpha[...,:3]).convert("RGB")
                except:
                    try:
                        fh = open("sample-errorfile.txt", "a+", encoding="utf-8")
                    except:
                        fh = open("sample-errorfile.txt", "w", encoding="utf-8")
                    fh.write(normal_path+"\n")
                    fh.close()
                    continue

                image, degradation = albedo, image

                # relighting
                if random.random() > 0.5:
                    envmap = None
                    if os.path.exists(envmap_path) == False:
                        envmap_path = image_path
                    self_envmap = read_envmap(envmap_path)
                    size = 256
                    h, w = self_envmap.shape[:2]
                    ratio = size/min(h, w)
                    self_envmap = cv2.resize(self_envmap, (int(ratio*w), int(ratio*h)))                
                    
                    h, w = self_envmap.shape[:2]
                    dh = h-min(h, w)
                    dw = w-min(h, w)
                    size = min(h, w)
                    
                    if self.enable_timer:
                        e_time = time.time()
                        timer["load_materials"].append(e_time-s_time)
                        s_time = time.time()

                    # 先滤波再 crop 的亮度小于先 crop 再滤波
                    # coeffs = spherical_harmonics_coeffs_v2(self_envmap*255, order=2, num_samples=100000)
                    # env_lighting = generate_spherical_image_v2(coeffs, self_envmap.shape[1], self_envmap.shape[0]) # TODO
                    # env_lighting = env_lighting[dh//2:dh//2+min(h, w), dw//2:dw//2+min(h, w)]

                    # self_envmap_crop = self_envmap[dh//2:dh//2+min(h, w), dw//2:dw//2+min(h, w)]
                    s_t = time.time()
                    self_envmap_crop = np.concatenate([self_envmap[:, -size//2:], self_envmap[:, :size//2]], axis=1)[:, ::-1]
                    coeffs_crop = spherical_harmonics_coeffs_v2(self_envmap_crop*255, order=2, num_samples=sh_num_steps)
                    env_lighting_crop = generate_spherical_image_v2(coeffs_crop, self_envmap_crop.shape[1], self_envmap_crop.shape[0]) # TODO
                    # cv2.imwrite("sample-lighting.png", np.concatenate([env_lighting_crop, self_envmap*255], axis=1))
                    # image.save("sample-image.png")
                    e_t = time.time()
                    # print("Shconverter: ", e_t-s_t, "sec.")

                    lighting_mask = gen_lighting_mask(grid_size=(4,4), target_size=(32,32))[..., np.newaxis]
                    env_lighting = cv2.resize(env_lighting_crop, (32,32))[...,[2,1,0]]
                    env_lighting1 = env_lighting*lighting_mask
                    env_lighting2 = env_lighting*(1.-lighting_mask)

                    if self.enable_timer:
                        e_time = time.time()
                        timer["prepare_LML1L2"].append(e_time-s_time)
                        s_time = time.time()

                    # lighting_mask = np.concatenate([ligradient_checkpointingghting_mask]*3, axis=-1)*255
                    # cv2.imwrite("sample-LML1L2.png", np.concatenate([env_lighting, lighting_mask, env_lighting1, env_lighting2], axis=1))

                    # degradade
                    degradation, degradation_lighting = self.image_degrader.ibr_image_degrade(
                        np.array(image)[...,::-1],
                        np.array(albedo)[...,::-1],
                        np.array(normal)[...,::-1],
                        envmap=envmap_path if random.random() > 0.2 else None,
                        sh_num_steps=sh_num_steps,
                        lighting_scale=random.uniform(1.25, 1.5),
                        hard_shadow_min_val=random.uniform(0.4,0.7),
                    )
                    # print("degradation_lighting:", degradation_lighting.shape, degradation_lighting.min(), degradation_lighting.max())
                    degradation_lighting = cv2.resize(degradation_lighting, (degradation.shape[1], degradation.shape[0]))
                    degradation_lighting = Image.fromarray(degradation_lighting[...,::-1])
                    degradation = Image.fromarray(degradation[...,::-1])

                if self.enable_timer:
                    e_time = time.time()
                    timer["degradation"].append(e_time-s_time)
                    s_time = time.time()
                
                images = [image, mask, degradation]
                # e_time = time.time()
                images, original_size_as_tuple, crop_coords_top_left = resize_and_crop(
                    images, bucket.image_width, bucket.image_height
                )

                images = [self.t_norm(self.t_toTensor(item)) for item in images]

                h, w = images[0].shape[1:]
                # degradation_lighting = self.t_norm(self.t_toTensor(degradation_lighting.resize((w,h))))

                image = images[0]
                mask = images[1] * .5 + .5
                degradation = images[2]
                mask = torch.mean(mask, dim=0, keepdim=True)
                image = image*mask
                degradation = degradation*mask

                # lighting as condition
                '''
                lighting_mask = torch.from_numpy(lighting_mask).float().permute(2,0,1)
                env_lighting = torch.from_numpy(env_lighting).float().permute(2,0,1)/127.5-1
                env_lighting1 = torch.from_numpy(env_lighting1).float().permute(2,0,1)/127.5-1
                env_lighting2 = torch.from_numpy(env_lighting2).float().permute(2,0,1)/127.5-1

                degradation = degradation*mask
                
                if self.enable_timer:
                    e_time = time.time()
                    timer["Aug&toTensor"].append(e_time-s_time)
                    s_time = time.time()
                '''

                sample = {
                    # "image", "text", "degradation", "envmap", "envmap1", "envmap2"
                    "image": image,  # --type=torch.Tensor --shape=(3,h,w) --range=(-1,1)
                    "mask": mask,    # --type=torch.Tensor --shape=(1,h,w) --range=(0,1)
                    "text": "uniformly lighting",#text,
                    "degradation": degradation,# --type=List<torch.Tensor> --item-shape=(3,h,w) --range=(-1,1)
                    '''
                    "lighting": env_lighting,
                    "lighting_mask": lighting_mask,
                    "lighting1": env_lighting1,
                    "lighting2": env_lighting2,
                    "degradation_lighting": degradation_lighting,
                    '''
                    "original_size_as_tuple": original_size_as_tuple,
                    "crop_coords_top_left": crop_coords_top_left,
                    "target_size_as_tuple": torch.tensor([bucket.image_height, bucket.image_width]),
                    "dataset":"IBR data"
                }
                if self.for_fusion:
                    sample["bg_image"] = image*(1.-mask)
                    sample["text"] = "harmonious, natural, photorealistic, seamless, homogeneous"

                # e_time = time.time()
                if self.enable_timer:
                    print("++ Yield data return time cost:")
                    print(timer)
                
                yield sample
                # samples.append(sample)
                # sample_returned += 1
               
                # yield samples

            """
            print("++++++++++ Read batch time cost")
            for key in timer:
                print("++ {}: {:.2f}ms".format(key, np.mean(np.array(timer[key]))*1000))
            """    
            # samples = self.make_batch(samples)
    
        # ++++++++++ Read batch time cost
        # ++ load_materials: 99.15ms
        # ++ prepare_LML1L2: 341.91ms # Sh_converter: 333 ms          -> 85
        #                                             -(10w->1w)> 80
        #                                             -(ctrcrop)> 333
        # ++ degradation: 946.99ms    # Sh_converter: 780 ms          -> 250
        #                                             -(10w->1w)> 500
        #                                             -(ctrcrop)> 540
        # ++ Aug&toTensor: 85.24ms

    def plotting(self, items):
        canvas = []
        canvas_lighting = []
        canvas_dlighting = []

        for item in items:    
            for b in range(item["edited_pixel_values"].shape[0]):
                sample = torch.cat([item["fg_mask_values"][b].expand(3,-1,-1)*2.-1., item["edited_pixel_values"][b], item["original_fg_values"][b]], dim=-1)
                sample = ((sample.permute(1,2,0).numpy()*.5+.5)*255.).astype(np.uint8)
                # for text in [item["prompt"][b], item["text"][b]]:
                #     if text is not None:
                #         paper = prompt2image(text, font_size=w//36, height=h//4, width=w) 
                #     canvas.append(paper)
                sample_lighting = torch.cat(
                    [item["lighting_values"][b], item["lighting_masks"][b].expand(3,-1,-1)*2.-1., item["lighting1_values"][b], item["lighting2_values"][b]], dim=-1
                )
                sample_lighting = ((sample_lighting.permute(1,2,0).numpy()*.5+.5)*255.).astype(np.uint8)
                # sample_dlighting = ((item["degradation_lighting"][b].permute(1,2,0).numpy()*.5+.5)*255.).astype(np.uint8)
                canvas.append(sample)
                canvas_lighting.append(sample_lighting)
                # canvas_dlighting.append(sample_dlighting)

        cv2.imwrite("sample-iter.png", np.concatenate(canvas, axis=0)[..., ::-1])
        cv2.imwrite("sample-iter-lighting.png", np.concatenate(canvas_lighting, axis=0)[..., ::-1])
        # cv2.imwrite("sample-iter-degradation-lighting.png", np.concatenate(canvas_dlighting, axis=0)[..., ::-1])

    def simple_plotting(self, item):
        canvas = []  
        for b in range(item["image"].shape[0]):
            sample = torch.cat([item["mask"][b].expand(3,-1,-1)*2.-1., item["image"][b], item["degradation"][b]], dim=-1)
            sample = ((sample.permute(1,2,0).numpy()*.5+.5)*255.).astype(np.uint8)
            canvas.append(sample)
        cv2.imwrite("sample-unilit.png", np.concatenate(canvas, axis=0)[..., ::-1])



def resize_and_crop(images, target_width, target_height):
    # gt, fg, bg, mask = images
    # assert image.size == mask.size, "图像和mask尺寸不一致。"
    # if gt.size != mask.size:
    #     print("!!!!!!!图像和mask尺寸不一致。", gt.size, mask.size)
        # image = image.rotate(-90, expand=True)
        # length = len(os.listdir('/mnt/bn/gongyuany/Experiments/No_matching/img/'))
        # image.save(f'/mnt/bn/gongyuany/Experiments/No_matching/img/{length}.jpg')
        # mask.save(f'/mnt/bn/gongyuany/Experiments/No_matching/mask/{length}.jpg')
    orig_width, orig_height = images[0].size
    
    # 计算缩放比例
    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height
    
    # 如果宽度或高度有一个已经符合目标尺寸，则不需要缩放
    # if width_ratio >= 1 and height_ratio >= 1:
    #     ratio = 1
    # else:
    # 选取缩放比例较大的一边进行resize
    if width_ratio > height_ratio:
        ratio = width_ratio
        new_width = target_width
        new_height = int(orig_height * ratio)
    elif width_ratio < height_ratio:
        ratio = height_ratio
        new_width = int(orig_width * ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = target_height
    
    # 缩放图像和mask
    images_resized = [image.resize((new_width, new_height)) for image in images]
    # gt_resized = gt.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # fg_resized = fg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # bg_resized = bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # mask_resized = mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
    original_size_as_tuple = torch.tensor([images[0].height, images[0].width])

    # 如果另一边超出目标尺寸，进行裁剪
    if new_width > target_width:
        delta_w = new_width - target_width
        left = random.randint(0, delta_w)
        right = left + target_width
        crop_coords_top_left = torch.tensor([0, left])
        # fg_cropped = fg_resized.crop((left, 0, right, new_height))
        # mask_cropped = mask_resized.crop((left, 0, right, new_height))
        images_cropped = [image.crop((left, 0, right, new_height)) for image in images_resized]
    elif new_height > target_height:
        delta_h = new_height - target_height
        lower = random.randint(0, delta_h)
        upper = lower + target_height
        crop_coords_top_left = torch.tensor([lower, 0])
        # image_cropped = image_resized.crop((0, lower, new_width, upper))
        # mask_cropped = mask_resized.crop((0, lower, new_width, upper))
        images_cropped = [image.crop((0, lower, new_width, upper)) for image in images_resized]
    else:
        crop_coords_top_left = torch.tensor([0, 0])
        images_cropped = images_resized
        # mask_cropped = mask_resized
    return images_cropped, original_size_as_tuple, crop_coords_top_left


if __name__ == "__main__":
    data_root = "/mnt/bn/dreamlight-bytenas-lq/dataset_dreamlight/"
    data_root = "/mnt/bn/dreamlight-lf2-42f98e68/datasets/dataset_dreamlight"
    data_root = "/mnt/bn/xiaowenpeng-bytenas-hl3/datasets/dataset_dreamlight"
    image_root= f"{data_root}/images"
    mask_root = f"{data_root}/masks"
    envmap_root= f"{data_root}/envmaps/difflight"
    meta_root = f"{data_root}/info_new_subset-of-switchlight"
    # eigen_root = f"{data_root}/eigens"
    # eigen_root = f"{data_root}/eigens-unet"
    switchlight_root = (f"{data_root}/eigens-switchlight", )
    hard_shadow_root = ["/mnt/bn/xiaowenpeng-bytenas-hl3/datasets/hard_shadows/光影笔刷PS"]

    dataset = DreamlightImageMultiscaleDataset(
        image_root, mask_root, envmap_root, meta_root, switchlight_root, hard_shadow_root,
        batch_size=1,
        bucket_override=[
            (704, 320), (320, 768), (896, 256), (640, 384), (576, 448), (832, 256),
            (512, 512), (448, 576), (384, 640), (384, 640), (1024, 256), (320, 704),
            (768, 320), (960, 256), (256, 1024), (256, 960), (256, 896), (256, 832)
        ],
        debug=["appending_buckets_tags"],
        bucket_ratio=1.5,
    )
    fbucket
    dataloader_dreamlight = DataLoader(dataset=dataset,
                                   batch_size=8,
                                   num_workers=0)
    # dataloader_dreamlight.sample()
    dataloader_iter_dreamlight = iter(dataloader_dreamlight)
    
    idx = 0
    while True:
        s_time = time.time()
        batch = next(dataloader_iter_dreamlight)
        e_time = time.time()
        dataset.simple_plotting(batch)
        print(idx, "Time cost:", e_time-s_time, "sec.", len(batch), batch["image"].shape, "\n\n")
        breakpoint()

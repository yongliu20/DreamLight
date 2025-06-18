import io
import os
import re
import html
import random
import urllib.parse as ul
import ftfy
import torch
import numpy as np
import importlib
from abc import abstractmethod
from base64 import b64decode
from typing import Any, Dict, Optional, Tuple, List
from bs4 import BeautifulSoup
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import json 
from .hdfs import listdir
from copy import deepcopy
from .coco_utils import coco_classes, thing_classes


def resolve_interpolation_mode(interpolation_type):
    if interpolation_type == "bilinear":
        interpolation_mode = F.InterpolationMode.BILINEAR
    elif interpolation_type == "bicubic":
        interpolation_mode = F.InterpolationMode.BICUBIC
    elif interpolation_type == "nearest":
        interpolation_mode = F.InterpolationMode.NEAREST
    elif interpolation_type == "lanczos":
        interpolation_mode = F.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" modes are `bilinear`, `bicubic`, `lanczos`, and `nearest`."
        )

    return interpolation_mode


class ImageDecoder:
    """
    Decode image from json dictionary. Return None if sample cannot be decoded to skip forward.
    """
    def __init__(self):
        # Avoid image too large warning messages.
        Image.MAX_IMAGE_PIXELS = 1000000000

    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        image_data = item.get("image_org") or item.get("image") or item.get("binary")
        if image_data is None:
            return None

        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            image_bytes = b64decode(image_data)
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                    image = image.convert("RGBA")
                    white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                    white.paste(image, mask=image.split()[3])
                    image = white
                else:
                    image = image.convert("RGB")
        except:
            image = Image.frombytes("RGB", (item["W"], item["H"]), image_bytes)
        return image


class MaskDecoder:
    """
    Decode mask from json dictionary. Return None if sample cannot be decoded to skip forward.
    """
    def __init__(self):
        # Avoid image too large warning messages.
        Image.MAX_IMAGE_PIXELS = 1000000000
    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        mask_data = item.get("masks")
        mask_data = [mask_data] if not isinstance(mask_data, list) else mask_data
        masks = []
        for data in mask_data:
            if data is None:
                masks.append(None)
                continue
            if isinstance(data, bytes):
                mask_bytes = data
            else:
                mask_bytes = b64decode(data)
            with Image.open(io.BytesIO(mask_bytes)) as mask:
                masks.append(mask.convert("L"))
        return masks
    
class SaliencyMaskDecoder:
    """
    Decode mask from json dictionary. Return None if sample cannot be decoded to skip forward.
    """
    def __init__(self):
        # Avoid image too large warning messages.
        Image.MAX_IMAGE_PIXELS = 1000000000
    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        data = item.get("saliency_mask", None)
        if data is None:
            return None
        if isinstance(data, bytes):
            mask_bytes = data
        else:
            mask_bytes = b64decode(data)
        try:
            with Image.open(io.BytesIO(mask_bytes)) as mask:
                mask = mask.convert("L")
        except:
            mask = Image.frombytes("L", (item["W"], item["H"]), mask_bytes)
        return mask

class GeneralImageDecoder(ImageDecoder):
    """
    Read image from hdfs data entry, usually is in bytes format
    """
    def __init__(self):
        # Avoid image too large warning messages.
        Image.MAX_IMAGE_PIXELS = 1000000000

    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        image_data = item.get("image_org") or item.get("image") or item.get("binary")
        if image_data is None:
            return None

        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            image_bytes = b64decode(image_data)

        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
        return image


# -----------------------------------------------------------------------------------------------

class ImagePredicate:
    """
    Check if image satifiy a certaion requirements.
    Return False if not satisfied and True if pass the check.

    Be sure to pass key-value pair when using
    """
    @abstractmethod
    def __call__(self, image: Image.Image, **kwargs) -> bool:
        raise NotImplementedError()


class ImageMultiPredicate(ImagePredicate):
    def __init__(self, predicates: List[ImagePredicate]):
        self.predicates = predicates

    def __call__(self, image: Image.Image, **kwargs) -> bool:
        for predicate in self.predicates:
            if not predicate(image, **kwargs):
                return False
        return True


class ImageWhiteBorderPredicate(ImagePredicate):
    """check if image has white border"""
    def _has_white_border(self, img: Image, border_width: int = 3):
        def has_vertical_border():
            width, height = img.size
            border_color = (255, 255, 255) # we check only white border for now
            for y in range(height):
                for bw in range(border_width):
                    # Check left border
                    if img.getpixel((bw, y)) != border_color:
                        return False
                    # Check right border
                    if img.getpixel((width - 1 - bw, y)) != border_color:
                        return False
            return True

        def has_horizontal_border():
            width, height = img.size
            border_color = (255, 255, 255)

            for x in range(width):
                for bw in range(border_width):
                    # Check top border
                    if img.getpixel((x, bw)) != border_color:
                        return False
                    # Check bottom border
                    if img.getpixel((x, height - 1 - bw)) != border_color:
                        return False
            return True
        return any([has_vertical_border(), has_horizontal_border()])

    def __call__(self, image: Image.Image, **kwargs) -> bool:
        return (not self._has_white_border(image, border_width=3))


class ImageSizePredicate(ImagePredicate):
    def __init__(self, min_size, debug=False):
        self.min_size = min_size
        self.debug = debug

    def __call__(self, image: Image.Image, **kwargs) -> bool:
        match = image.size[0] >= self.min_size and \
                image.size[1] >= self.min_size

        if not match and self.debug:
            print(f"Skip sample. Image size: {image.size}. Required size: {self.min_size}.")

        return match


class ImageBucketResolutionPredicate(ImagePredicate):
    def __call__(self, image: Image.Image, bucket: Any, **kwargs) -> bool:
        if image.size[0] < bucket.image_width or image.size[1] < bucket.image_height:
            return False
        return True


class ImageAestheticPredicate(ImagePredicate):
    def __init__(self, aes_thed=0):
        self.aes_thed = aes_thed

    def __call__(self, image: Image.Image, content: dict, **kwargs) -> bool:
        return ("aesthetic" not in content) or (content["aesthetic"] >= self.aes_thed)


class ImageMultiSourceSmallResolutionPredicate(ImagePredicate):
    """
    In this class, we deliberately want small resolution images
    So we will return False if a image is larger than a given size,
    because a large image does not satisfy the requirement
    """
    def __init__(self, small_image_size=384):
        self.small_image_size=small_image_size

    def __call__(self, image: Image.Image, **kwargs):
        if image.width >= self.small_image_size and image.height >= self.small_image_size and \
                image.height*image.width >= self.small_image_size * self.small_image_size and \
                0.5 <= image.width/image.height <= 2:
            return False
        if image.width * image.height <= 128 * 128:
            # we also don't want the image to be too small
            return False
        return True


# -----------------------------------------------------------------------------------------------


class TextDecoder:
    """
    Decode text from json dictionary.
    Return None or raise exception if sample cannot be decoded to skip forward.
    """

    def __init__(self, prefix_prob: float = 0, prefix_content: str = ""):
        self.prefix_prob = prefix_prob
        self.prefix_content = prefix_content

    @abstractmethod
    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        keys = ["caption", "blip2_opt", "blip2_t5",
                "blip2_opt_text", "blip2_t5_text"]
        captions = [item.get(key, None) for key in keys]
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        caption = random.choice(captions) if len(captions) > 0 else None
        return caption

    def __call__(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        caption = self._get_caption(item, index_file, *args, **kwargs)
        if caption is not None and caption != "":
            if random.random() < self.prefix_prob:
                caption = self.prefix_content + caption
        return caption


class GeneralTextDecoder(TextDecoder):
    """
    Example item dict:
    {
        "blip2_t5": "a black and white photo of a woman holding a large owl",
        "blip2_opt": "a black and white photo of a woman holding an owl",
    }
    """
    def __init__(
        self,
        prefix_prob: float = 0,
        prefix_content: str = "",
        keys: List[str] = None):
        super(GeneralTextDecoder, self).__init__(prefix_prob, prefix_content)
        self.keys = keys
    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        default_keys = ["caption", "text", "blip2_opt", "blip2_t5",
                "blip2_opt_text", "blip2_t5_text"]
        if self.keys is None:
            self.keys = default_keys
        captions = [item.get(key, None) for key in self.keys]
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        caption = random.choice(captions) if len(captions) > 0 else None
        return caption


class TuchongFilterV3TextDecoder(TextDecoder):
    """
    Example item dict:
    {
        "title_original": "a black and white photo of a woman holding a large owl",
        "title": "a black and white photo of a woman holding an owl",
        "remark_original": "black and white,woman,owl,",
    }
    """
    def _tuchong_split_zh_en(self, input_text: str) -> Tuple[str, str]:
        """
        split text into chinese and english by iterating through each character
        and check they are letters or chinese, TODO: clean it up if necessary
        """
        index = []
        for iid, text in enumerate(input_text):
            if not text.isascii():
                index.append(iid)
        if len(index) > 0:
            zh_text = input_text[index[0]: index[-1]]
            en_text = input_text[:index[0]] + input_text[index[-1] + 1:]
            return (zh_text, en_text) if len(en_text) > 5 else (zh_text, "")
        return ("", input_text)

    def _tuchong_text_filtered(self, text_original: str, text: str) -> Tuple[str, str]:
        """
        tuchong text processing, select one from two given resources,
        favor the first one TODO: clean it up if necessary
        """
        if text_original != "":
            return self._tuchong_split_zh_en(text_original)
        if text != "":
            return self._tuchong_split_zh_en(text)
        return ("", "")

    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:

        title_original = item.get("title_original", "")
        title = item.get("title", "")
        remark = item.get("remark_original", "")
        remark_original = item.get("remark", "")

        res = []
        _, res_en = self._tuchong_text_filtered(title_original, title)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)
        _, res_en = self._tuchong_text_filtered(remark, remark_original)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)
        if len(res) > 0:
            return random.choice(res)
        return None


class HumanTextDecoder(TextDecoder):
    """
    Example item dict:
    {
        "all_caltions": {
            "blip2_t5": "a black and white photo of a woman holding a large owl",
            "blip2_opt": "a black and white photo of a woman holding an owl",
            "LLaVA_short_tags": "A woman is standing in the snow, holding a white owl on her arm"
            "WDTag": {
                "1girl": 0.9923139214515686,
                "bird": 0.9679114818572998,
                "monochrome": 0.9564778208732605
            },
        }
    }
    """
    def _waifu_wd_tag(self, text: Optional[str], item: Dict[str, Any]) -> Optional[str]:
        wd_tags = item.get("wd-v1-4-vit-tagger-v2",
                           item.get("all_captions", {}).get("WDTag", None))
        if wd_tags is not None:
            # compatible with different ways of save WD tags
            if isinstance(wd_tags, list):
                wd_tags = list(wd_tags[0].keys())
            elif isinstance(wd_tags, dict):
                wd_tags = list(wd_tags.keys())
            elif isinstance(wd_tags, str):
                wd_tags = [w.strip(", ") for w in wd_tags.split(",")]
            else:
                return text

            wd_tags = [w for w in wd_tags if (
                'boy' not in w and 'girl' not in w)]

            # we hard limit the tag number to use, use at most 3 wd tags a time
            subset_size = min(random.randint(0, len(wd_tags)),
                              random.choice([1, 2, 3]))
            if subset_size:
                random.shuffle(wd_tags)
                # sample from tags
                wd_subset = wd_tags[:subset_size]
                if 'monochrome' in wd_tags and 'monochrome' not in wd_subset:
                    wd_subset.insert(0, 'monochrome')
                if "greyscale" in wd_tags and 'greyscale' not in wd_subset:
                    wd_subset.insert(0, 'greyscale')

                final_wd_set = [wd for wd in wd_subset if wd not in text]
                wd_sentence = ", ".join(final_wd_set)

                if text is None or text == "":
                    text = wd_sentence
                else:
                    # 3 cases: front, end, and not use
                    use_level = random.randint(0, 2)
                    if use_level == 0:
                        text = text.strip(",. ") + ", " + wd_sentence
                    elif use_level == 1:
                        text = wd_sentence + ", " + text
        return text

    def _style_by_source_(self,
                          text: Optional[str],
                          item: Optional[dict] = None,
                          index_file: Optional[str] = None,) -> Optional[str]:
        if text is None:
            return None

        def get_style_tag_by_resource():
            if index_file is not None and "midjourney" in index_file:
                return "in kodak film style"
            if item is not None:
                source_from_item = item.get("source", None)
                if "midjourney" in source_from_item or "mj" in source_from_item:
                    return "in kodak film style"
                elif source_from_item is not None:
                    return "masterpiece photography"
            return False

        style_tag = get_style_tag_by_resource()
        if style_tag is not None:
            if random.random() < 0.6:
                text = f"{text.strip('., ')}, {style_tag}"
        return text

    def _process_mj_original_text(self, item: Dict[str, Any]) -> Optional[str]:
        """
        we want to get rid of mj caption's original style prompts
        """
        def check_mj_camera_info(cap):
            style_words = ['kodak', 'canon', 'fuji']
            return any(sw in cap for sw in style_words)

        if "caption_mj" in item.keys() and len(item.get("caption_mj", "")) > 0:
            mj_cap = item.get("caption_mj")
            cap_parts = [s.strip(", ") for s in mj_cap.split(",") if not check_mj_camera_info(s) ]
            mj_cap = ", ".join(cap_parts)
            return mj_cap
        return None

    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        # Randomly choose captions
        captions = [cap for cap in item["all_captions"].values()]
        captions.append(self._process_mj_original_text(item))
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        if len(captions) > 0:
            caption = random.choice(captions)
            # add wd-tags, 66% chance using wd-tag
            caption = self._waifu_wd_tag(text=caption, item=item)
            # add style words by data resource, 60% chance using it
            caption = self._style_by_source_(text=caption, item=item, index_file=index_file)
            # TODO: refine the asian race tag adding logic, but it is what used for now
            if index_file is not None and 'asian' in index_file and random.random()<0.7:
                caption = 'asian, ' + caption
            return caption
        return None


class AnimeTextDecoder(TextDecoder):
    """
    Example item dict:
    {
        "blip2_opt": "a cartoon of a man in a boxing ring with a woman"
        "comic_score_v2": [0.1, 0.02, 0.3, 0.007, ...]
    }
    """
    idx2tags_v3 = {
        0: ['anime'],  # 占位 备用
        1: ['japanese anime', 'celluloid'],  # 赛璐璐
        2: ['japanese anime', 'semi-thick painting'],  # 半厚涂
        3: ['japanese anime', 'retro manga'],  # 复古昭和
        4: ['japanese anime', 'woodblock prints'],  # 日式版画
        5: ['japanese anime', 'chibi'],  # Q版人物
        6: ['japanese anime', 'acrylic painting'],  # 厚涂
        7: ['anime', 'fantasy', 'realistic'],  # 梦幻超写实
        8: ['american comics', '2D'],  # 2D美漫
        9: ['american comics', 'acrylic painting'],  # 厚涂美漫
        10: ['american cartoon'],  # 美式怪诞
        11: ['american comics', 'retro comics'],  # 复古美漫
        12: ['pixar', '3D'],  # 卡通3D
        13: ['chinese painting', 'watercolor painting'],  # 水彩/墨
        14: ['chinese anime', 'ancient chinese'],  # 古风仙侠
        15: ['anime', 'webtoon'],  # 现代都市
        16: ['anime'],  # 动漫其他
    }

    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        # Get style tags
        if random.random() < 0.6:
            style_score = (
                item.get('comic_score_v3_array', None) or
                item.get('comic_score_v3', None) or
                item.get('comic_score_v2', None) or
                [0]
            )
            style_index = np.argmax(style_score)
            style_tags = self.idx2tags_v3.get(style_index, [])
            random.shuffle(style_tags)
        else:
            style_tags = []

        # Get blip caption
        keys = ["blip2_opt", "blip2_t5", "blip2_opt_text", "blip2_t5_text"]
        captions = [item.get(key, None) for key in keys]
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        caption = random.choice(captions)

        # Combine both
        caption = ', '.join([caption] + style_tags)
        return caption


class CGTextDecoder(TextDecoder):
    idx2tags_3dcg_v1 = {
        0: ['3D', 'CG', 'pixar'],
        1: ['3D', 'CG', 'realistic'],
        2: ['3D', 'CG', 'acrylic painting'],
    }

    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        # Get style tags
        if random.random() < 0.6 and "3dcg_score_v1" in item.keys():
            style_score = item['3dcg_score_v1']
            style_index = np.argmax(style_score)
            style_tags = self.idx2tags_3dcg_v1.get(style_index, [])
            random.shuffle(style_tags)
        else:
            style_tags = []

        # Get blip caption
        keys = ["blip2_opt", "blip2_t5", "blip2_opt_text", "blip2_t5_text"]
        captions = [item.get(key, None) for key in keys]
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        caption = random.choice(captions)

        # Combine both
        caption = ', '.join([caption] + style_tags)
        return caption


class MidjourneyTextDecoder(TextDecoder):
    """
    Example item dict:
    {
        "blip2_t5": "a black and white photo of a woman holding a large owl",
        "blip2_opt": "a black and white photo of a woman holding an owl",
    }
    """
    def _get_caption(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        keys = ["clean_prompts", "file_text_text", "blip2_opt",
                "blip2_t5", "blip2_opt_text", "blip2_t5_text"]
        captions = [item.get(key, None) for key in keys]
        captions = [c for c in captions if isinstance(c, str) and len(c) > 0]
        caption = random.choice(captions) if len(captions) > 0 else None
        return caption


class MultiSourceTextDecoder:
    """
    This class allows use of different text decoder for different index_file path:
    Example:

        decoder = text_decoder=MultiSourceTextDecoder(
            default=GeneralTextDecoder()
            mapping={
                "3dcg": CGTextDecoder(),
                "unsplash": UnsplashTextDecoder(),
                "realhuman2_blip2": HumanTextDecoder(),
            },
        )
    """
    def __init__(
        self,
        mapping: Dict[str, TextDecoder],
        default: TextDecoder = GeneralTextDecoder(),
    ):
        self.mapping = mapping
        self.default = default

    def __call__(
        self,
        item: Dict[str, Any],
        index_file: Optional[str] = None,
        *args,
        **kwargs
        ) -> Optional[str]:
        selected_decoder = self.default
        if index_file:
            for path, decoder in self.mapping.items():
                # If current index_path matches any part of the path in the mapping,
                # select the first matched decoder.
                if index_file in path or path in index_file:
                    selected_decoder = decoder
                    break

        return selected_decoder(*args, item, index_file, **kwargs)


# -----------------------------------------------------------------------------------------------


class TextCleaner:
    """
    Clear up a caption with strange/improper contents
    """
    bad_punct_regex = re.compile(
        r'[' + '#®•©™&@·º½¾¿¡§~' + '\)' + '\(' + '\]' + '\[' + '\}' + '\{' + '\|' + '\\' + '\/' + '\*' + r']{1,}')

    def __call__(self, text):
        # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        caption = re.sub('<br>', ' ', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)

        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(
            r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(self.bad_punct_regex, r' ', caption)
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(
            r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        # j2d1a2a...
        caption = re.sub(
            r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()


# -----------------------------------------------------------------------------------------------


class RandomCrop:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: torch.Tensor):
        if self.factor == 1:
            return image

        _, H, W = image.shape
        h = round(H / self.factor)
        w = round(W / self.factor)
        x = int(random.random() * (W - w))
        y = int(random.random() * (H - h))
        return image[:, y:y+h, x:x+w]


class MultiSourceTextDecoder_aesFinetune(TextDecoder):
    """
    Decode txt for aes_model,  there may be many primary labels for diffierent source.
    """

    def __call__(self, item, index_file, primary_label="", rare_token="", camera_exif=False,
                 caption_keys=["caption", "text", "blip2_opt", "blip2_t5", "tag", "title", "desc",
                               "blip2_opt_text", "blip2_t5_text", 'deepdanbooru_caption', "file_text_text",
                               "llava_short_text", "clean_prompts"]):

        if "tuchong" in index_file.lower():
            res = self.fileterd_tuchong(item).replace(",,", ",")
            if rare_token != "":
                res = f"{rare_token},{res}"
            return res.replace(",,", ",")

        res = self.get_caption(item, caption_keys)
        if res is None:
            return None
            # if camera_exif:
        #     exif = self.get_camera_exif(item)
        #     res =  f"{caption},{exif}".replace(",," , ",")

        if primary_label != "":
            if primary_label == "commic":
                label = self.get_style_tag(item)
                res = f"{label},{res}".replace(",,", ",")
            else:
                res = f"{primary_label},{res}".replace(",,", ",")

        if rare_token != "":
            res = f"{rare_token},{res}"

        return res.replace(",,", ",")

    def fileterd_tuchong(self, item):
        title_original = item.get("title_original", "")
        title = item.get("title", "")
        keywords_original = item.get("keywords_original", "")
        keywords = item.get("keywords", "")
        remark_original = item.get("remark_original", "")
        remark = item.get("remark", "")

        res = []
        _, res_en = self.tuchong_text_filtered(title_original, title)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        # 0426 temporarily remove data only with keywords
        # _, res_en = self.tuchong_text_filtered(keywords_original, keywords)
        # if res_en != "" and len(res_en) > 3:
        #     res.append(res_en)

        _, res_en = self.tuchong_text_filtered(remark, remark_original)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        blip2_opt = item.get("blip2_opt_text", "")
        res.append(blip2_opt)

        # --------------get sorted keywords--------------
        sorted_keywords = self.get_sorted_keywords(item)
        res.append(sorted_keywords)

        res = [i for i in res if i is not None and i != ""]

        if len(res) > 0:
            return random.choice(res)
        return ""

    def tuchong_text_filtered(self, text_original: str, text: str) -> Tuple[str, str]:
        """
        tuchong text processing, select one from two given resources, favor the first one TODO: clean it up if necessary
        """
        if text_original != "":
            return self.tuchong_split_zh_en(text_original)
        if text != "":
            return self.tuchong_split_zh_en(text)
        return ("", "")

    def tuchong_split_zh_en(self, input_text: str) -> Tuple[str, str]:
        """
        split text into chinese and english by iterating through each character
        and check they are letters or chinese, TODO: clean it up if necessary
        """
        index = []
        for iid, text in enumerate(input_text):
            if not text.isascii():
                index.append(iid)
        if len(index) > 0:
            zh_text = input_text[index[0]: index[-1]]
            en_text = input_text[:index[0]] + input_text[index[-1] + 1:]
            return (zh_text, en_text) if len(en_text) > 5 else (zh_text, "")
        return ("", input_text)

    def get_camera_exif(self, item):
        camera_len = item.get('camera_len', {})
        res = ""
        for k in camera_len:
            res = res + f",{k} {camera_len[k]}"
        return res

    def get_caption(self, item, caption_keys):
        """
        random choice
        """
        texts = []
        for key in caption_keys:
            texts.append(item.get(key, ""))
        texts = [i for i in texts if i != ""]
        texts = [i for i in texts if i is not None]
        if texts == []:
            return None
        return random.choice(texts)

    def get_sorted_keywords(self, item):
        sorted_keywords = item.get("keywords_sorted_clip_text", None)
        if sorted_keywords is None:
            return ""
        sorted_keywords = sorted_keywords.split(",")[:10]
        return ",".join(sorted_keywords)

    def get_comic_label(self, item):
        return ""

    def get_style_tag(self, data_item):
        idx2tags_v3 = {
            0: ['anime'],  # 占位 备用
            1: ['japanese anime, celluloid'],  # 赛璐璐
            2: ['japanese anime, semi-thick painting'],  # 半厚涂
            3: ['japanese anime, retro manga'],  # 复古昭和
            4: ['japanese anime, woodblock prints'],  # 日式版画
            5: ['japanese anime, chibi'],  # Q版人物
            6: ['japanese anime, acrylic painting'],  # 厚涂
            7: ['anime, fantasy, realistic'],  # 梦幻超写实
            8: ['american comics, 2D'],  # 2D美漫
            9: ['american comics, acrylic painting'],  # 厚涂美漫
            10: ['american cartoon'],  # 美式怪诞
            11: ['american comics, retro comics'],  # 复古美漫
            12: ['pixar, 3D'],  # 卡通3D
            13: ['chinese painting, watercolor painting'],  # 水彩/墨
            14: ['chinese anime, ancient chinese'],  # 古风仙侠
            15: ['anime, webtoon'],  # 现代都市
            16: ['anime'],  # 动漫其他
            17: ['']  # 非动漫
        }
        style_score = data_item.get('comic_score_v3_array',
                                    data_item.get('comic_score_v3', data_item.get('comic_score_v2', [0])))
        if len(style_score) == 1:
            style_idx = 0
        else:
            style_idx = np.argmax(style_score) + 1
        if style_idx < 17:
            return random.choice(idx2tags_v3[style_idx])
        return ''


def resize_crop(image, image_height, image_width, use_resize_random_crop=False, mask=None):
    aspect_ratio = image_width / image_height
    image_aspect_ratio = image.width / image.height
    if image_aspect_ratio >= aspect_ratio:
        image_resize_h = image_height
        image_resize_w = int(round(image_height * (image.width / image.height)))
        crop_top_coord = 0
        if use_resize_random_crop:
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            crop_left_coord = (image_resize_w - image_width) // 2
    else:
        image_resize_w = image_width
        image_resize_h = int(round(image_width * (image.height / image.width)))
        crop_left_coord = 0
        if use_resize_random_crop:
            crop_top_coord = random.randint(0, image_resize_h - image_height)
        else:
            crop_top_coord = (image_resize_h - image_height) // 2
    image = F.resize(image, size=[image_resize_h, image_resize_w],
                        interpolation=InterpolationMode.LANCZOS)
    image = F.crop(image, crop_top_coord, crop_left_coord, image_height,
                    image_width)
    crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
    if mask is not None:
        mask = F.resize(mask, size=[image_resize_h, image_resize_w],
                            interpolation=InterpolationMode.NEAREST)
        mask = F.crop(mask, crop_top_coord, crop_left_coord, image_height,
                        image_width)
        return image, mask, crop_coords_top_left
    return image, crop_coords_top_left



###  shiqi data
class TextDecoder2:
    """
    Decode text from json dictionary. Return None if sample cannot be decoded to skip forward.
    """
    def __call__(self, item: Dict[str, Any], *args, **kwargs) -> Optional[str]:
        texts = []

        text = item.get(random.choice(["blip2_opt", "blip2_t5"]))
        if text is not None:
            # Add main text caption.
            texts.append(text)
        else:
            # If text is not available, randomly sample from metadata.
            potential_keys = [k for k in item.keys() if k in ["caption", "title", "text", "desc"]]
            if len(potential_keys) > 0:
                texts.append(item.get(random.choice(potential_keys)))

        # Primary labels describes style such as "3D CG" or "anime"
        primary_labels = item.get("primary_labels")
        if isinstance(primary_labels, list) and len(primary_labels) > 0:
            texts.append(random.choice(primary_labels))

        # Secondary labels describes the primary label, eg "hyperrealism"
        secondary_labels = item.get("secondary_labels")
        if isinstance(secondary_labels, list) and len(secondary_labels) > 0:
            texts.append(random.choice(secondary_labels))

        # danbooru tags such as "boy, fantasy, etc". Add with a 50% chance.
        danbooru_tags = item.get("deepdanbooru_caption")
        if danbooru_tags is not None and len(danbooru_tags) > 0 and random.random() > 0.5:
            # randomly sample tags within danbooru
            danbooru_tags = danbooru_tags.split(",")
            sample_size = random.randint(0, len(danbooru_tags))
            danbooru_tags = random.sample(danbooru_tags, sample_size)
            texts += danbooru_tags

        random.shuffle(texts)
        text_delimiter = ", "
        text = text_delimiter.join(texts)
        return text


class MultiSourceTextDecoder2(TextDecoder):
    """
    Decode text from json dictionary. Return None if sample cannot be decoded to skip forward.
    """
    def multi_tag_text_making(self, item: Dict[str, Any]) -> Optional[str]:
        """
        customized a text description from multiple sources including labels, blip2-captions, tags
        """
        text_danbooru = item.get("deepdanbooru_caption", "") if random.randint(0,1) == 1 else ""

        text = item.get(random.choice(["blip2_opt", "blip2_t5"]), "")
        if text == "":
            # in case no blip2 caption available, find origianl text descriptors
            text_keys = []
            for key in item.keys():
                for potential_key in ["caption", "title", "text", "desc"]:
                    if potential_key in key:
                        if key != "text_length": # this if is for coyo dataset
                            text_keys.append(key)
            if len(text_keys) >= 1:
                text = item.get(random.choice(text_keys), "")

        if text == "" or "tag" in item.keys():
            # try if it is unsplash or pixbay
            text_tag = item.get("tag", "")
            if len(text_tag) > 0 and random.randint(0,1)==1:
                text += " "  + text_tag

        # danbooru tags such as "boy, fantasy, etc". Add with a 50% chance.
        if random.randint(0, 1):
            text_danbooru = item.get("deepdanbooru_caption")
            if text_danbooru is not None and len(text_danbooru):
                # randomly sample tags within danbooru
                text_danbooru = text_danbooru.split(",")
                subset_size = random.randint(0, len(text_danbooru))
                if subset_size:
                    # randomly shuffle tags
                    random.shuffle(text_danbooru)
                    # sample from tags
                    subset = text_danbooru[:subset_size]
                    text = text + ", ".join(subset)

        # add camera lens info for images from unsplash and pixabay photography data
        if "camera_len" in item.keys():
            camera_dict = item["camera_len"]
            camera_sent = ""
            for camera_value in camera_dict.values():
                if "none" not in str(camera_value).lower() and len(str(camera_value)) > 2:
                    camera_sent += f"{camera_value}, "
            camera_sent = camera_sent.strip(", ")
            if len(camera_sent) > 0:
                text = text + ", camera info: " + camera_sent

        return text

    def append_keywords_by_source(self, text, source):
        """
        customized text by using some keywords to describe a image with certain level of aesthetics
        """
        if source.split("_")[0] in ["huaban"]:
            text = "asian aesthetic, " + text
        if "cgsociety_artstation_wallhaven_v2" in source:
            text = "art aesthetic score 0.8+, " + text
        elif "3d_alldata_round2" in source or \
                "cgsociety_main" in source or \
                    "high_aesthetics_round1" in source:
            text = "art aesthetic score 0.7, " + text
        elif "3dcg_human_selected" in source:
            text = "art hand-selected, " + text
        return text
    def get_caption(self, item , keys = None , index_file = None  ):
        """
        random choice
        """
        if keys is None :
            caption_keys = ["caption", "text", "blip2_opt", "blip2_t5", "tag", "title", "desc", "tags",
                            "blip2_opt_text", "blip2_t5_text", 'deepdanbooru_caption', "file_text_text",
                            "llava_short_text", "clean_prompts"]
        else:
            caption_keys = keys
        texts = []
        for key in caption_keys:
            texts.append(item.get(key, ""))
        texts = [i for i in texts if i != ""]
        texts = [i for i in texts if i is not None]
        if texts == []:
            return None
        text =  random.choice(texts)
        #remove the space character for tusou data
        if index_file is not None and "tusou" in index_file:
            text = text.replace(" "  , '')
        if isinstance(text , float):
            text = ''
        return text
    def __call__(self, item: Dict[str, Any], index_file: Optional[str] = None, *args, **kwargs) -> Optional[str]:
        use_zh =  kwargs.pop("use_zh" , False)
        use_en =  kwargs.pop("use_en" , False)
        use_customed_tuchong  = kwargs.pop("use_customed_tuchong" , False)
        # decode caption for taisu 
        if index_file is not None and "taisu" in index_file: 
            if use_zh:
                text = self.get_taisu_cn_en_caption(item , True)
                return text 
            if use_en :
                text =  self.get_taisu_cn_en_caption(item , False)
                return text 
        # decode caption for wukong 
        if index_file is not None and "tusou" in index_file: 
            if use_zh:
                text = self.get_tusou_cn_en_caption(item , True)
                return text
            else:
                return item.get("blip_opt_cn_text" , "")
        if index_file is not None and "wukong" in index_file: 
            if use_zh:
                text = self.get_wukong_cn_en_caption(item , True)
                return text 
            if use_en :
                text =  self.get_wukong_cn_en_caption(item , False)
                return text 
        if index_file is not None and "digicol.dpm_cn" in index_file:
            if use_zh:
                text = self.get_dpm_cn_en_caption(item , True )
                return text
            if use_en:
                text  = self.get_dpm_cn_en_caption(item , False)
                return text
        if index_file is not None and "vcg_cn" in index_file:
            if use_zh:
                text = self.get_vcg_cn_en_caption(item , True )
                return text
            if use_en:
                text  = self.get_vcg_cn_en_caption(item , False)
                return text

        if index_file is not None and "Zero" in index_file:
            if use_zh:
                text = self.get_Zero_cn_en_caption(item , True)
                return text 
            if use_en :
                text =  self.get_Zero_cn_en_caption(item , False)
                return text 
        ####this is a customed  tuchong text decoder, only for double language
        if index_file is not None and "tuchong" in index_file and use_customed_tuchong :
            if use_zh:
                text = item.get(   "blip2_opt_cn_text" ,  None    )
                return text
            else:
                text = item.get(   "blip2_opt_text" ,  None    )
                return text 
            
        if index_file is not None and use_zh:
            return  self.get_caption(item , keys= ["title_cn", 'title_cn_text' , 'remark_cn_text' , "caption_zh" , "web_caption_cn" , "generated_caption_cn"] , index_file = index_file )
        if index_file is not None and use_en:
            return self.get_caption(item , keys= ["caption"] )
        if index_file is not None and 'tuchong' in index_file:
            text = self.fileterd_tuchong( item )
        elif index_file is not None and ('laion' in index_file and 'translated' in index_file):
            text = self.laion_3b_translated( item )
        elif index_file is not None and "aes_0707" in index_file:
            text = self.get_caption(item)
        elif index_file is not None and 'datacomp-1b' in index_file.lower():
            text = self.get_DataComp_1B_caption(item)
        else:
            text = self.multi_tag_text_making( item )
        return text

    def get_DataComp_1B_caption(self, item: Dict[str, Any]) -> Optional[str]:
        text = item.get("text", None)
        return text
    def get_taisu_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            caption = item.get("web_caption_cn" , None )
            if caption is not None :
                return caption
            caption = item.get("generated_caption_cn" , None )
            if caption is not None :
                return caption
            caption = item.get("blip2_opt_text" , None )
            return caption 
        else:
            extra_meta_translate_cn_to_en_dict = json.loads( item["extra_meta_translate_cn_to_en_dict"] )
            caption = extra_meta_translate_cn_to_en_dict.get("web_caption_cn_en_text" , None )
            if caption is not None :
                return caption
            caption = extra_meta_translate_cn_to_en_dict.get("generated_caption_cn_en_text" , None )
            if caption is not None :
                return caption
            caption = item.get("blip2_opt_cn_text" , None )
    def get_wukong_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            caption = item.get("title_cn" , None )
            return caption
        else:
            extra_meta_translate_cn_to_en_dict = json.loads( item["extra_meta_translate_cn_to_en_dict"] )
            caption = extra_meta_translate_cn_to_en_dict.get("title_cn_en_text" , None )
            return caption
    def get_tusou_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            caption = item.get("caption_zh" , item.get("caption" , None ) )
            if caption is not None :
                caption = caption.replace(" "  , '')
            return caption
        else:
            raise NotImplementedError
    def get_dpm_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            keys = ["name" , "category_name" , "period" , "blip2_opt_cn_text" ]
            texts = []
            for k in keys:
                if item.get(k , None ) is not None :
                    texts.append( item.get(k , None ) )
            random.shuffle(texts)
            return "， ".join(texts)
        else:
            return item.get( "blip2_opt_text"  , None )
    def get_vcg_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            return item.get("img_detail_title" , None )
        else:
            return item.get( "blip2_opt_text"  , None )
    def get_Zero_cn_en_caption(self , item: Dict[str, Any] ,  ZH = True) -> Optional[str]:
        if ZH:
            caption = item.get("image_query" , None )
            if isinstance(caption , list) and len(caption) > 0:
                return random.choice(  caption )
            return None
        else:
            try: 
                extra_meta_translate_cn_to_en_dict = json.loads( item["extra_meta_translate_cn_to_en_dict"] )
                caption = extra_meta_translate_cn_to_en_dict.get("image_query_en_array" , None )
                if isinstance(caption , list) and len(caption) > 0:
                    return random.choice(  caption )
            except:
                return None

    def laion_3b_translated(self, item: Dict[str, Any]) -> Optional[str]:
        text = item.get("ENG TEXT", None)
        if text is None:
            # actually should not use this text, this text is not in english
            text = item.get("TEXT", None)
        return text

    def tuchong_split_zh_en(self, input_text: str) -> Tuple[str, str]:
        """
        split text into chinese and english by iterating through each character
        and check they are letters or chinese, TODO: clean it up if necessary
        """
        index = []
        for iid, text in enumerate(input_text):
            if not text.isascii():
                index.append(iid)
        if len(index) > 0:
            zh_text = input_text[index[0]: index[-1]]
            en_text = input_text[:index[0]] + input_text[index[-1] + 1:]
            return (zh_text, en_text) if len(en_text) > 5 else (zh_text, "")
        return ("", input_text)

    def tuchong_text_filtered(self, text_original: str, text: str) -> Tuple[str, str]:
        """
        tuchong text processing, select one from two given resources, favor the first one TODO: clean it up if necessary
        """
        if text_original != "":
            return self.tuchong_split_zh_en(text_original)
        if text != "":
            return self.tuchong_split_zh_en(text)
        return ("", "")

    def fileterd_tuchong(self, item: Dict[str, Any]) -> Optional[str]:
        """
        ugly tuchong text processing, TODO: clean it up if necessary
        """
        title_original    = item.get("title_original", "")
        title             = item.get("title", "")
        keywords_original = item.get("keywords_original", "")
        keywords          = item.get("keywords", "")
        remark            = item.get("remark_original", "")
        remark_original   = item.get("remark", "")

        res = []
        _, res_en = self.tuchong_text_filtered(title_original, title)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        # 0426 temporarily remove data only with keywords
        # _, res_en = self.tuchong_text_filtered(keywords_original, keywords)
        # if res_en != "" and len(res_en) > 3:
        #     res.append(res_en)

        _, res_en = self.tuchong_text_filtered(remark, remark_original)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        if len(res) > 0:
            return random.choice(res)
        return None

class TextCleaner2:
    bad_punct_regex = re.compile(
        r'[' + '#®•©™&@·º½¾¿¡§~' + '\)' + '\(' + '\]' + '\[' + '\}' + '\{' + '\|' + '\\' + '\/' + '\*' + r']{1,}')
    def __init__(self, handle_cn=False):
        self.handle_cn = handle_cn
    def __call__(self, text):
        # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        caption = re.sub('<br>', ' ', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text
        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)
        if not self.handle_cn:
            # 31C0—31EF CJK Strokes
            # 31F0—31FF Katakana Phonetic Extensions
            # 3200—32FF Enclosed CJK Letters and Months
            # 3300—33FF CJK Compatibility
            # 3400—4DBF CJK Unified Ideographs Extension A
            # 4DC0—4DFF Yijing Hexagram Symbols
            # 4E00—9FFF CJK Unified Ideographs
            caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
            caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
            caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
            caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
            caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
            caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
            caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
            #######################################################
        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            # noqa
            '-', caption)
        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)
        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)
        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)
        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)
        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)
        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "
        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)
        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231
        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)
        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)
        return caption.strip()

class DataFilterDefault():
    def __init__(self, aes_thed=0):
        self.aes_thed = aes_thed
    def __call__(self, image, text, *args, **kwargs):
        content = args[1]
        if image.width * image.height <= 128 * 128:
            return True
        if "aesthetic" in content and content['aesthetic'] < self.aes_thed:
            return True
        return False

class DataFilterMultiSourceSmall(DataFilterDefault):
    def __call__(self, image, text, *args, **kwargs):
        content = args[1]
        if image.width >= 384 and image.height >= 384 and image.height*image.width >= 200704 and 0.5<=image.width/image.height<=2:
            return True
        if image.width * image.height <= 128 * 128:
            return True
        if "aesthetic" in content and content['aesthetic'] < self.aes_thed:
            return True
        return False

class DataFilterMultiSourceSmall1024(DataFilterDefault):
    def __call__(self, image, text, *args, **kwargs):
        content = args[1]
        if image.width >= 768 and image.height >= 768 and image.height*image.width >= 589824 and 0.5<=image.width/image.height<=2:
            return True
        if image.width * image.height <= 128 * 128:
            return True
        if "aesthetic" in content and content['aesthetic'] < self.aes_thed:
            return True
        return False

class DataFilterMultiSourceSmall256(DataFilterDefault):
    def __call__(self, image, text, *args, **kwargs):
        if image.width >= 192 and image.height >= 192 and image.height*image.width >= 50176 and 0.5<=image.width/image.height<=2:
            return True
        if image.width * image.height <= 128 * 128:
            return True
        return False


class DataFilterByImageSize(DataFilterDefault):
    def __call__(self, image, text, *args, **kwargs):
        bucket = args[0]
        content = args[1]
        if image.size[0] < bucket.image_width or image.size[1] < bucket.image_height:
            return True
        if image.width * image.height <= 128 * 128:
            return True
        if "aesthetic" in content and content['aesthetic'] < self.aes_thed:
            return True
        return False


def get_aes_tag_for_commic(item , cn = False , index = False) : 
    '''
        光影-明暗对比-chiaroscuro
        重心-视觉重心- visual weight / visual focus
        质感-细节材质- textured / detailed / exquisite / high detail / hyper quality
        构图-视觉排布-composition / visual arrangement
        情感-情绪表达-Emotional
        色彩-颜色搭配-aesthetic color / good color combination / gorgeous color scheme
        颜值-好看潮流- lovely face / good looking
    '''
    aes_dict = {
        5: 'chiaroscuro', # 光影
        7: 'visual focus', # 重心
        4: 'textured, detailed', # 质感-细节材质
        6: 'visual arrangement', # 构图-视觉排布
        8: 'emotional', # 情感-情绪表达
        3: 'aesthetic color', # 色彩-颜色搭配
        9: 'good looking', # 颜值-好看潮流

        0: 'normal quality', # < 3.5
        1: 'high quality', # 3.5
        2: 'best quality' # 4/5
    }
    aes_dict_cn = {
        5: '光影', # 光影
        7: '重心', # 重心
        4: '质感', # 质感-细节材质
        6: '构图', # 构图-视觉排布
        8: '情感', # 情感-情绪表达
        3: '色彩', # 色彩-颜色搭配
        9: '颜值', # 颜值-好看潮流

        0: '标准', # < 3.5
        1: '优质', # 3.5
        2: '最优' # 4/5
    }

    aes_tag = ''
    aes_tag_cn = ''
    if cn :
        if item.get('fine_grained'):
            aes_index = item.get('fine_grained',[])
            for idx in aes_index:
                aes_tag_cn += '， ' + aes_dict_cn.get(idx, '')
        return aes_tag_cn.strip('，')
    else:
        if item.get('fine_grained'):
            aes_index = item.get('fine_grained',[])
            for idx in aes_index:
                aes_tag += ', ' + aes_dict.get(idx, '')
        return aes_tag.strip(',')

def get_aes_tag_for_photo(item , cn):
    anno_data = item.get("anno_data" , "" )
    if anno_data == "":
        return anno_data
    aes_tag_map = {
        "quality":["normal quality" ,"high quality" ,'best quality' ],
        "chiaroscuro" : ["" , "chiaroscuro"] , 
        "visual_focus" : ["" , 'visual focus'] , 
        "aesthetic_color": ["" , 'aesthetic color'],
        "textured_detailed": ["" , 'textured, detailed'],
        "visual_arrangement": ["" , 'visual arrangement'],
        "emotional" : ["" , 'emotional'],
    }
    aes_tag_cn_map = {
        "quality":["标准" ,"优质" ,'最优' ],
        "chiaroscuro" : ["" , "光影"], 
        "visual_focus" : ["" , '重心'] , 
        "aesthetic_color": ["" , '色彩'],
        "textured_detailed": ["" , '质感'],
        "visual_arrangement": ["" , '构图'],
        "emotional" : ["" , '情感'],
    }
    aes_tag = []
    if cn : 
        for k in anno_data.keys():
            if k in aes_tag_cn_map:
                aes_tag.append( aes_tag_cn_map[k][ anno_data[k]  ] )
    else:
        for k in anno_data.keys():
            if k in aes_tag_map:
                aes_tag.append( aes_tag_map[k][ anno_data[k]  ] )
    return ", ".join(aes_tag )

    
class MultiSourceTextDecoder_aesFinetune2(TextDecoder):
    """
    Decode txt for aes_model,  there may be many primary labels for diffierent source.
    """
    def __call__ (self , item ,index_file, primary_label = "", rare_token = "",camera_exif = False , use_style_score = False , caption_keys = None  , cn = False , use_aes_tag = False):

        #  to get the caption for tuchong data
        if "tuchong" in index_file.lower():
            res  = self.fileterd_tuchong(item).replace(",," , ",")
            if rare_token != "":
                res = f"{rare_token},{res}"
            return res.replace(",," , ",")

        if caption_keys is None :
            caption_keys = ["caption" ,"text", "blip2_opt" , "blip2_t5" , "tag", "title", "desc","tags","blip2_opt_text","blip2_t5_text" ,
                             'deepdanbooru_caption' , "file_text_text" , "llava_short_text" , "clean_prompts"] 
        res = self.get_caption(item , caption_keys)
        if res is None :
            return None 
        # if camera_exif:
        #     exif = self.get_camera_exif(item)
        #     res =  f"{caption},{exif}".replace(",," , ",")

        if primary_label != "":
            if primary_label == "commic":
                label = self.get_style_tag(item , index_file , cn)
                aes_tag = get_aes_tag_for_commic(item , cn)
                if label != '' and use_aes_tag:
                    if random.random() > 0.5 : 
                        res =  f"{label},{res},{aes_tag}".replace(",," , ",")
                        res = res.strip(",")
                    else:
                        res =  f"{res},{label},{aes_tag}".replace(",," , ",")
                        res = res.strip(",")
                elif label != '' and not use_aes_tag:
                    if random.random() > 0.5 : 
                        res =  f"{label},{res}".replace(",," , ",")
                        res = res.strip(",")
                    else:
                        res =  f"{res},{label}".replace(",," , ",")
                        res = res.strip(",")
            else:
                res = f"{primary_label}, {res}, {aes_tag}".replace(",," , ",")
        if use_style_score : 
            label = self.get_mj_style_tag(item)
            if label != "":
                if random.random() > 0.5 :
                    res =  f"{label},{res}".replace(",," , ",")
                else:
                    res =  f"{res},{label}".replace(",," , ",")
            else:
                pass
        if rare_token != "":
            res = f"{rare_token},{res}"
        if use_aes_tag : 
            aes_tag = get_aes_tag_for_photo(item , cn )
            res= res.strip(".").strip("。").strip(",").strip("，")
            res += ", " + aes_tag.strip(",")

        return res.replace(",," , ",")
        
    def fileterd_tuchong(self , item):
        title_original    = item.get("title_original", "")
        title             = item.get("title", "")
        keywords_original = item.get("keywords_original", "")
        keywords          = item.get("keywords", "")
        remark_original           = item.get("remark_original", "")
        remark   = item.get("remark", "")

        res = []
        _, res_en = self.tuchong_text_filtered(title_original, title)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        # 0426 temporarily remove data only with keywords
        # _, res_en = self.tuchong_text_filtered(keywords_original, keywords)
        # if res_en != "" and len(res_en) > 3:
        #     res.append(res_en)

        _, res_en = self.tuchong_text_filtered(remark, remark_original)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        blip2_opt = item.get("blip2_opt_text" , "")
        res.append(blip2_opt)

        #--------------get sorted keywords--------------
        sorted_keywords = self.get_sorted_keywords(item)
        res.append(sorted_keywords)

        res = [i for i in res if i!= ""]
        
        if len(res) > 0:
            return random.choice(res)
        return ""
    def tuchong_text_filtered(self, text_original: str, text: str) -> Tuple[str, str]:
        """
        tuchong text processing, select one from two given resources, favor the first one TODO: clean it up if necessary
        """
        if text_original != "":
            return self.tuchong_split_zh_en(text_original)
        if text != "":
            return self.tuchong_split_zh_en(text)
        return ("", "")
    def tuchong_split_zh_en(self, input_text: str) -> Tuple[str, str]:
        """
        split text into chinese and english by iterating through each character
        and check they are letters or chinese, TODO: clean it up if necessary
        """
        index = []
        for iid, text in enumerate(input_text):
            if not text.isascii():
                index.append(iid)
        if len(index) > 0:
            zh_text = input_text[index[0]: index[-1]]
            en_text = input_text[:index[0]] + input_text[index[-1] + 1:]
            return (zh_text, en_text) if len(en_text) > 5 else (zh_text, "")
        return ("", input_text)

    def get_camera_exif(self , item):
        camera_len = item.get('camera_len' , {})
        res = ""
        for k in camera_len:
            res = res + f",{k} {camera_len[k]}"
        return res
    def get_caption(self, item , caption_keys):
        """
        random choice
        """
        texts  = []
        for key in caption_keys:
            texts.append( item.get(key  , "")  )
        texts = [i for i in texts if i != ""]
        texts = [i for i in texts if i is not  None ]
        if texts == []:
            return None 
        return random.choice(texts)
    def get_sorted_keywords(self , item):
        sorted_keywords = item.get("keywords_sorted_clip_text" , None)
        if sorted_keywords is None:
            return ""
        sorted_keywords = sorted_keywords.split(",")[:10]
        return ",".join(sorted_keywords)
    def get_comic_label(self , item):
        return ""
    def get_style_tag(self, data_item , index_file , cn ):
        idx2tags_v3 = {
                    0:['anime'], # 占位 备用
                    1:['japanese anime, celluloid'], # 赛璐璐
                    2:['japanese anime, semi-thick painting'], # 半厚涂
                    3:['japanese anime, retro manga'], # 复古昭和
                    4:['japanese anime, woodblock prints'], # 日式版画
                    5:['japanese anime, chibi'], # Q版人物
                    6:['japanese anime, acrylic painting'], # 厚涂
                    7:['anime, fantasy, realistic'], # 梦幻超写实
                    8:['american comics, 2D'], # 2D美漫
                    9:['american comics, acrylic painting'], # 厚涂美漫
                    10:['american cartoon'], # 美式怪诞
                    11:['american comics, retro comics'], # 复古美漫
                    12:['pixar, 3D'], #  卡通3D
                    13:['chinese painting, watercolor painting'], #  水彩/墨
                    14:['chinese anime, ancient chinese'], # 古风仙侠
                    15:['anime, webtoon'], # 现代都市
                    16:['anime'], # 动漫其他
                    17:[''] # 非动漫
                        }
        idx2tags_v3_cn = {
                    0:['日漫'], # 占位 备用
                    1:['日漫， 赛璐璐'], # 赛璐璐
                    2:['日漫， 半厚涂'], # 半厚涂
                    3:['日漫， 复古昭和'], # 复古昭和
                    4:['日漫， 日式版画'], # 日式版画
                    5:['日漫， Q版人物'], # Q版人物
                    6:['日漫， 厚涂'], # 厚涂
                    7:['日漫， 梦幻超写实'], # 梦幻超写实
                    8:['2D美漫'], # 2D美漫
                    9:['厚涂美漫'], # 厚涂美漫
                    10:['美式怪诞'], # 美式怪诞
                    11:['复古美漫'], # 复古美漫
                    12:['卡通3D'], #  卡通3D
                    13:['水彩， 水墨'], #  水彩/墨
                    14:['古风， 仙侠'], # 古风仙侠
                    15:['漫画， 现代都市'], # 现代都市
                    16:['动漫，其他'], # 动漫其他
                    17:[''] # 非动漫
                        }
        idx2tags_3dcg_v1 = {
            0: ['3D, CG, pixar'],
            1: ['3D, CG, realistic'],
            2: ['3D, CG, acrylic painting'],
            3: ['']
        }
        #3d,CG 
        if index_file.split('/')[-2] in ['3dcg_human_selected_0606_buckets_ci_bucket' , "artstation_v4_filtered_0607_ci_bucket" , "3dcg_by_likes_v2_buckets_ci_bucket"]:
            style_score_3dcg = data_item.get('3dcg_score_v1')
            if style_score_3dcg is not None:
                style_idx_3dcg = np.argmax(style_score_3dcg)
                if style_idx_3dcg < 4:
                    if random.random() < 0.1 :
                        return ""
                    return random.choice(idx2tags_3dcg_v1[style_idx_3dcg])

        style_score = data_item.get('comic_score_v4',data_item.get('comic_score_v3_array', data_item.get('comic_score_v3',data_item.get('comic_score_v2',[0]))))
        if len(style_score) == 1:
            style_idx = 0
        else:
            style_idx = np.argmax(style_score) + 1
        if style_idx < 17:
            ####random dropout 
            if random.random() < 0.1 :
                return ""
            if not cn:
                return random.choice(idx2tags_v3[style_idx])
            else:
                return random.choice(idx2tags_v3_cn[style_idx])
        return ''
    def get_mj_style_tag(self ,   item ):
        idx2tags = {
            0 : ["movie Cartoon 3D"] , 
            1 : ["movie hyper-realistic 3D"] , 
            2 : ["movie, others"] , 
            3 : ["anime"] , 
            4 : ["American animation"] , 
            5 : ["Chinese animation"] , 
            6 : ["anime, others"] , 
            7 : ["traditional art,"], 
            8 : ["Avant-garde art"] , 
            9 : [ "punk art"] , 
            10 : [ "Realistic cartoon painting style" ] , 
            11 : ["Avant-garde art, others"] , 
            12 : [""],
            13 : [""],
            14 : [""],
        }
        try:
            style_idx = np.argmax(item["style_score_v1_array"])
            assert style_idx < 15
            if random.random() < 0.8 :
                return random.choice( idx2tags[style_idx] )
            return ""
        except:
            style_idx = 14
            return random.choice( idx2tags[style_idx] )

def has_logo(data_item):
    #返回值为True，不含水印
    #返回值为False，滤除当前图片
    
    ocr_res = data_item['ocr_v3']
    if not isinstance(ocr_res , dict):
        try: 
            rsp_json = eval(ocr_res)
        except:
            return True 
    else:
        rsp_json = ocr_res
    if len(rsp_json['words']) == 0:
        return True

    # 读取裁剪前的图片长/宽
    width, height =0, 0
    for subdict in rsp_json['extra']:
        if subdict['key'] == "height":
            height = int(subdict['value'])
        elif subdict['key'] == "width":
            width = int(subdict['value'])

    # 读取文字坐标点和内容，并进行判断
    textlis = []
    for word in rsp_json['words']:
        points = word['det_points_abs']
        textname = word['text']
        textlis.append(textname)
        xcur, ycur = [], []
        for point in points:
            xcur.append(point['x'])
            ycur.append(point['y'])
        xmin, xmax = min(xcur), max(xcur)
        ymin, ymax = min(ycur), max(ycur)

        # 判断循环水印
        count = len(textlis) - len(set(textlis))
        if count >= 4:
            return False
        textname = textname.lower()

        # 判断中心文字
        if xmin > width / 8 and xmax < 7 * width / 8 and ymin > height / 8 and ymax < 7 * height / 8:
            target_chars = [
                'com', 'net', 'www', 'cn',
                '网',
                '红书',   # 小红书
                '公司',   # xx公司
                '厂',   # xx厂
                '同城',   # 58同城
                '东', '京东', 'jd',    # 京东
                'PChouse',
                '设备',

            ]
            for char in target_chars:
                if char in textname:
                    return False
        
        # 判断底部超过1/8的未被剪裁掉的水印文字
        if ymax > 7 * height / 8 and ymin < 7 * height / 8:
            target_chars = [
                'com', 'cn',
                '天下', 'fang',
                '大众点评',
                '孔夫子',
                '网',
                '赶集',
                '社区',
                '公司',
                '本地宝',
                '搜狐',
                '淘',
                '爱奇艺', '爱奇', '奇艺',
                '经验', 'bai', 'du', 'baidu',
                '频道',
                '发布',
                '新闻',
                '二手',
                '@',

            ]
            for char in target_chars:
                if char in textname:
                    return False
        # 全图只要出现就干掉的文字
        target_chars = [
            '新华', '华网', '新华网',
            '光明网', '明网',
            '中新网', '新网',
            '新闻',
            'fang', '房天下',
            '京东', '专营', '正版', '官方', '旗舰店', '正图', '版书', '润鼎', '发票', 
        ]
        for char in target_chars:
            if char in textname:
                return False
    return True

def check_mj_camera_info(s):
    cameras = ['kodak', 'canon', 'fuji']
    for c in cameras:
        if c in s.lower():
            return True
    return False
def process_mj_original_text(item: Dict[str, Any]) -> Optional[str]:
    if "caption_mj" in item.keys() and len(item.get("caption_mj", "")) > 0:
        mj_cap = item.get("caption_mj")

        cap_parts = [s.strip(", ") for s in mj_cap.split(",") if not check_mj_camera_info(s) ]
        mj_cap = ", ".join(cap_parts)
        return mj_cap
    return None

def waifu_wd_tag(item: Dict[str, Any], text: Optional[str]) -> Optional[str]:
    wd_tags = item.get("wd-v1-4-vit-tagger-v2",  None)
    if wd_tags is None:
        wd_tags = item.get("all_captions", {}).get("WDTag", None)

    if wd_tags is not None :
        if isinstance(wd_tags, list):
            wd_tags = list(wd_tags[0].keys())
        elif isinstance(wd_tags, dict):
            wd_tags = list(wd_tags.keys())
        elif isinstance(wd_tags, str):
            wd_tags = [w.strip(", ") for w in wd_tags.split(",")]
        else:
            return text 

        wd_tags = [w for w in wd_tags if ('boy' not in w and 'girl' not in w)]

        # we hard limit the tag number to use
        subset_size = min( random.randint(0, len(wd_tags)), random.choice([1, 2, 3]) )
        if subset_size:
            random.shuffle(wd_tags)
            # sample from tags
            wd_subset = wd_tags[:subset_size]
            if 'monochrome' in wd_tags and 'monochrome' not in wd_subset:
                wd_subset.insert(0, 'monochrome')
            if "greyscale" in wd_tags and 'monochrome' not in wd_subset:
                wd_subset.insert(0, 'greyscale')

            final_wd_set = []
            for wd in wd_subset:
                if wd not in text:
                    final_wd_set.append(wd)

            wd_sentence = ", ".join(final_wd_set)

            if text is None or text=="":
                text = wd_sentence
            else:
                use_level = random.randint(0, 2)
                if use_level==0:
                    text = text.strip(",. ") + ", " + wd_sentence
                elif use_level==1:
                    text = wd_sentence + ", " + text
                else:
                    text = text
    return text 


class FaceTextDecoderV4:

    def __call__(self, item: Dict[str, Any], index_file=None, *args, **kwargs) -> Optional[str]:

        rare_token = kwargs.get("rare_token" , "")
        
        all_captions = item.get("all_captions", None)
        use_aes_tag = kwargs.get("use_aes_tag" , False)
        if all_captions is None:
            return None 
        
        possible_keys = [
            'blip2_t5', 
            'blip2_opt',
            'Vicuna_LLaVA_short_tags_summary0', 
            'Vicuna_LLaVA_short_tags_summary1', 
            'Vicuna_LLaVA_short_tags_summary2', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 3 words: ', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 5 words: ', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 8 words: '
        ]
        possible_texts = []
        for pk in possible_keys:
            cur_text = all_captions.get(pk, "")
            if len(cur_text) > 0:
                possible_texts.append(cur_text)
        
        # get mj original caption
        mj_cap = process_mj_original_text(item)
        if mj_cap is not None:
            possible_texts.append(mj_cap)

        if len(possible_texts) > 0:
            text = random.choice(possible_texts) 
        else:
            return None

        text = text.strip("'").strip('"').strip("., ")
        if use_aes_tag:
            text = self._add_detail_aes_tags( text , item ,   False)
        # text = waifu_wd_tag( item=item, text=text )
        #text = style_by_source_v2(text=text, item=item, index_file=kwargs.get("index_file", None) )  , normal do not need this 

        race = item.get( "race" , "")
        if race != "" and race != "unknown" and random.random()<0.5:
            race = race + ", "
        else:
            race = ""

        
        return rare_token+ race  + text
    def _add_detail_aes_tags(self,text : str ,  item: Dict[str, Any] , cn : bool = False) -> str:
        #aes tag example: ['3.5', '光影', '质感']
        aes_tags = item.get('aes_tags',[])
        aes_map = {
            '不合格': 'normal quality',
            '3.5': 'high quality',
            '4/5': 'best quality',
            '光影': 'chiaroscuro',
            '重心': 'visual focus',
            '质感': 'textured, detailed',
            '色彩': 'aesthetic color',
            '情感': 'emotional',
            '颜值': 'good looking',
            '构图': 'visual arrangement',
        }
        aes_cn_map = {
            '不合格': '标准',
            '3.5': '优质',
            '4/5': '最优',
            '光影': '光影',
            '重心': '重心',
            '质感': '质感',
            '色彩': '色彩',
            '情感': '情感',
            '颜值': '颜值',
            '构图': '构图',
        }

        if cn :
            aes_words = []
            for tag in aes_tags:
                if tag in aes_cn_map:
                    aes_words.append(aes_cn_map[tag])
                else:
                    # print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
                    pass
            aes_words = "， ".join( aes_words )
            return text + "， "  + aes_words
        else:
            aes_words = []
            for tag in aes_tags:
                if tag in aes_map:
                    aes_words.append(aes_map[tag])
                else:
                    print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
            aes_words = ", ".join( aes_words )
            return text + ", "  + aes_words
            
def _get_aes_idx_for_human( item: Dict[str , Any]   ):
    """
    this fuction will get the aes_tag list.
    """
    aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '情感' ,'构图' ]
    aes_idx = [0] * len(aes_tag_order)
    aes_tags = item.get('aes_tags',[])
    for i ,aes_tag in enumerate( aes_tag_order ):
        if aes_tag in aes_tags:
            aes_idx[i] = 2* i  + 1 
        else:
            aes_idx[i] = 2 * i
    return torch.tensor( aes_idx )


def _get_aes_idx_for_commic( item: Dict[str , Any]   ):
    """
    this fuction will get the aes_tag list.
    """
    aes_dict = {
        5: '光影', # 光影
        7: '重心', # 重心
        4: '质感', # 质感-细节材质
        6: '构图', # 构图-视觉排布
        8: '情感', # 情感-情绪表达
        3: '色彩', # 色彩-颜色搭配
        9: 'good looking', # 颜值-好看潮流

        0: '不合格', # < 3.5
        1: '3.5', # 3.5
        2: '4/5' # 4/5
    }
    aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '情感' ,'构图' ]
    aes_idx = [0] * len(aes_tag_order)
    aes_tags = []
    if item.get('fine_grained'):
            aes_index = item.get('fine_grained',[])
            for idx in aes_index:
                aes_tags.append( aes_dict.get(idx, '') )
    for i ,aes_tag in enumerate( aes_tag_order ):
        if aes_tag in aes_tags:
            aes_idx[i] = 2* i  + 1 
        else:
            aes_idx[i] = 2 * i
    return torch.tensor( aes_idx )

def _get_aes_idx_for_photo( item: Dict[str , Any]   ):
    """
    this fuction will get the aes_tag list.
    """
    aes_tag_map = {
        "quality":['不合格' ,'3.5' ,'4/5' ],
        "chiaroscuro" : ["" , '光影'] , 
        "visual_focus" : ["" , '重心'] , 
        "aesthetic_color": ["" , '色彩'],
        "textured_detailed": ["" , '质感'],
        "visual_arrangement": ["" , '构图'],
        "emotional" : ["" , '情感'],
    }
    aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '情感' ,'构图' ]
    aes_idx = [0] * len(aes_tag_order)
    aes_tags = []

    anno_data = item.get("anno_data" , {} )
    for k in anno_data.keys():
            if k in aes_tag_map:
                aes_tags.append( aes_tag_map[k][ anno_data[k]  ] )

    for i ,aes_tag in enumerate( aes_tag_order ):
        if aes_tag in aes_tags:
            aes_idx[i] = 2* i  + 1 
        else:
            aes_idx[i] = 2 * i
    return torch.tensor( aes_idx )
def _get_aes_idx_for_random():
    aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '情感' ,'构图' ]
    aes_idx = [0] * len(aes_tag_order)
    for i ,_ in enumerate( aes_tag_order ):
        if random.random() >=0.5:
            aes_idx[i] = 2* i  + 1
        else:
            aes_idx[i] = 2 * i
    return torch.tensor( aes_idx )
    
class FaceTextDecoderV4_CN:
    def __call__(self, item: Dict[str, Any], index_file=None, *args, **kwargs) -> Optional[str]:

        rare_token = kwargs.get("rare_token" , "")
        
        all_captions = item.get("all_captions_cn", None)
        use_aes_tag = kwargs.get("use_aes_tag" , False) 
        if all_captions is None:
            return None 
        
        possible_keys = [
            'blip2_opt_cn', 
            'blip2_t5_cn',
            "LLaVA_short_tags_cn"  , 
            "Vicuna_LLaVA_short_tags_summary0_cn" , 
            'Vicuna_LLaVA_short_tags_summary1_cn' , 
            'Vicuna_LLaVA_short_tags_summary2_cn' , 
            "Vicuna_LLaVA_short_tags_Summarize the subject in 3 words: _cn" , 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 5 words: _cn',
            'Vicuna_LLaVA_short_tags_Summarize the subject in 8 words: _cn' , 
        ]
        possible_texts = []
        for pk in possible_keys:
            cur_text = all_captions.get(pk, "")
            if len(cur_text) > 0:
                possible_texts.append(cur_text)
        
        # get mj original caption
        mj_cap = process_mj_original_text(item)
        if mj_cap is not None:
            possible_texts.append(mj_cap)

        if len(possible_texts) > 0:
            text = random.choice(possible_texts) 
        else:
            return None

        text = text.strip("'").strip('"').strip("., ")
        if use_aes_tag : 
            text = self._add_detail_aes_tags( text ,  item ,  True)
        # text = waifu_wd_tag( item=item, text=text )
        #text = style_by_source_v2(text=text, item=item, index_file=kwargs.get("index_file", None) )  , normal do not need this 
        race_mapping = {
            'white' : "白人" , 
            'asian' : "亚洲人" , 
            "black" : "黑人" ,
            'indian' : "印度人" , 
            'west asian' : "东亚人" , 

        }

        race = item.get( "race" , "")
        if race != "" and race != "unknown" and random.random()<0.5:
            race = race_mapping[race] + "， "
        else:
            race = ""

        
        return rare_token+ race  + text
    
    def _add_detail_aes_tags(self,text : str ,  item: Dict[str, Any] , cn : bool = False) -> str:
        #aes tag example: ['3.5', '光影', '质感']
        aes_tags = item.get('aes_tags',[])
        if aes_tags == []:
            return text
        aes_map = {
            '不合格': 'normal quality',
            '3.5': 'high quality',
            '4/5': 'best quality',
            '光影': 'chiaroscuro',
            '重心': 'visual focus',
            '质感': 'textured, detailed',
            '色彩': 'aesthetic color',
            '情感': 'emotional',
            '颜值': 'good looking',
            '构图': 'visual arrangement',
        }
        aes_cn_map = {
            '不合格': '标准',
            '3.5': '优质',
            '4/5': '最优',
            '光影': '光影',
            '重心': '重心',
            '质感': '质感',
            '色彩': '色彩',
            '情感': '情感',
            '颜值': '颜值',
            '构图': '构图',
        }

        if cn :
            aes_words = []
            for tag in aes_tags:
                if tag in aes_cn_map:
                    aes_words.append(aes_cn_map[tag])
                else:
                    print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
            aes_words = "， ".join( aes_words )
            return text.strip("。").strip("，") + "， "  + aes_words
        else:
            aes_words = []
            for tag in aes_tags:
                if tag in aes_map:
                    aes_words.append(aes_map[tag])
                else:
                    print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
            aes_words = ", ".join( aes_words )
            return text.strip(",").strip(".") + ", "  + aes_words

# shiqi develop v1.3
# -----------------------------------------------------------------------------------------------
class ImageWatermarkSizePredicate(ImageSizePredicate):
    """
    This class  will filter the watermark based on OCR.
    """
    def __init__(self , min_size , debug = False):
        super().__init__(min_size , debug)
    def __call__(self ,  image: Image.Image, **kwargs) -> bool:
        match = super().__call__(  image , **kwargs ) 
        watermark = self.has_watermark(kwargs.get( "content" , dict() ))
        if not watermark and self.debug :
            print(f"Skip this image because of watermark")
        return match and  watermark 

    def has_watermark( self ,   item: Dict[str, Any]   ):
        """
        If the image does not have watermark , this function will return True , else False.
        """
        if 'ocr_v3' not in item or item["ocr_v3"] is None : 
            return True
        ocr_res = item['ocr_v3']
        if not isinstance(ocr_res , dict):
            try: 
                rsp_json = eval(ocr_res)
            except:
                return True 
        else:
            rsp_json = ocr_res
        if len(rsp_json['words']) == 0:
            return True

        #load image width and height
        width, height =0, 0
        for subdict in rsp_json['extra']:
            if subdict['key'] == "height":
                height = int(subdict['value'])
            elif subdict['key'] == "width":
                width = int(subdict['value'])
        #load the text and (x , y)
        textlis = []
        for word in rsp_json['words']:
            points = word['det_points_abs']
            textname = word['text']
            textlis.append(textname)
            xcur, ycur = [], []
            for point in points:
                xcur.append(point['x'])
                ycur.append(point['y'])
            xmin, xmax = min(xcur), max(xcur)
            ymin, ymax = min(ycur), max(ycur)

            # If the text content is duplicated, we consider it to be a watermark
            count = len(textlis) - len(set(textlis))
            if count >= 4:
                return False
            textname = textname.lower()

            if xmin > width / 8 and xmax < 7 * width / 8 and ymin > height / 8 and ymax < 7 * height / 8:
                target_chars = [
                    'com', 'net', 'www', 'cn',
                    '网',
                    '红书', 
                    '公司',  
                    '厂',   
                    '同城',   
                    '东', '京东', 'jd',  
                    'PChouse',
                    '设备',

                ]
                for char in target_chars:
                    if char in textname:
                        return False
            #if the watermark text at the bottom exceeds 1/8 that has not been cropped
            if ymax > 7 * height / 8 and ymin < 7 * height / 8:
                target_chars = [
                    'com', 'cn','天下', 'fang','大众点评','孔夫子',
                    '网','赶集','社区','公司','本地宝',
                    '搜狐','淘','爱奇艺', '爱奇', '奇艺',
                    '经验', 'bai', 'du', 'baidu','频道',
                    '发布','新闻','二手','@',
                ]
                for char in target_chars:
                    if char in textname:
                        return False
            # skip this item if these text  
            target_chars = [
                '新华', '华网', '新华网','光明网', '明网',
                '中新网', '新网','新闻','fang', '房天下',
                '京东', '专营', '正版', '官方', '旗舰店', '正图', '版书', '润鼎', '发票', 
                ]
            for char in target_chars:
                if char in textname:
                    return False
        return True 
        
class AestheticDataSource:
    def __init__(self ,**kwargs ):
        self.__add_data__()
    @abstractmethod
    def __add_data__(self ):
        """
        We can define all data source with different config here.
        And the data structure must comply with the following:
        self.paths = {
            "data_source_name":{
                "path":[
                    "HDFS path"
                ],
                "rare_token":"" ,  #it will be placed at the beginning of the sentence.
                "camera_exif": False, #if True, the camera exif will be placed at the end of the sentence.
                "proportion": 0.25 #it represents the proportion of this data source in the entire aesthetic data, the sum of all proportion be equal to 1.
                "rare_idx" : None # the index of current data source, it will be mapped as a embedding.
                "caption_keys": None # It specifies the available caption key.
            }
        }
        """
        raise NotImplementedError
    def return_data_source(self , index_file : str ):
        """
        Given a specific file path, this function returns the data source.
        """
        for ds in self.paths:
            for path in self.paths[ds]["paths"]:
                if path in index_file:
                    return ds

        print('This is an unknown data path!!!')
        raise RuntimeError

    def reassign_data(self):
        """
        The data proportion is very important for  finetuning.
        This function can reassign the final data paths based on the "proportion" of each data source.
        """
        self.get_every_data_source_num()
        assert abs(sum([ self.paths[data_source]["proportion"] for data_source in self.paths ]) -1 ) < 0.000000000002 # Occasionally addition overflow occurs in python, so this value cannot be strictly equal to 0.
        total_data_num =  max( [  (self.paths[data_source]["num"]/self.paths[data_source]["proportion"]) for data_source in self.paths ]) 

        for data_source in self.paths:
            num =self.paths[data_source]["num"] 

            current_num = total_data_num * self.paths[data_source]["proportion"]
            self.paths[data_source]["paths"]  = int(current_num /self.paths[data_source]["num"] ) * self.paths[data_source]["paths"]
            self.paths[data_source]["times"] = int( current_num /self.paths[data_source]["num"] )
            self.paths[data_source]["num"] = int( current_num /self.paths[data_source]["num"] ) * self.paths[data_source]["num"]

        print("===============================>>")
        total_data_num = sum([ self.paths[data_source]["num"] for data_source in self.paths  ])
        print(f"dataset total  num is : {str(total_data_num)}")
        print("------------------------------->>")
        for data_source in self.paths:
            proportion = self.paths[data_source]["num"] / total_data_num
            num = self.paths[data_source]["num"]
            info_times = str(self.paths[data_source]["times"])
            info_files_num = str(self.paths[data_source]["files_num"])
            print(f"data source : {data_source}, num : {str(num)} , proportion : {proportion} , times : {info_times} , file_num: {info_files_num} ")

        return  self.repeat_paths()

    def get_every_data_source_num(self):
        """
        This function will calculate the number of each data source.
        """
        for data_source in self.paths:
            source_num = 0
            for path in self.paths[data_source]["paths"]:
                num = self.get_single_path_num(path,data_source)
                source_num += num
            self.paths[data_source]["num"] = source_num
    def get_single_path_num(self , path , data_source):
        res = 0
        files = listdir(path)
        if "all_files" not in self.paths[data_source]:
            self.paths[data_source]["all_files"] = []
        if "files_num" not in self.paths[data_source]:
            self.paths[data_source]["files_num"] = 0
        self.paths[data_source]["files_num"] += len(files)
        self.paths[data_source]["all_files"] = self.paths[data_source]["all_files"]  + deepcopy(files)
        files = [i for i in files if "index" in i]
        for filepath in files:
                # Parse name, example:
                #  filepath:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196.index"
                #  filename:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196"
                #  basename:  "2_19_256-896_00002_00196"
                filename, _ = os.path.splitext(filepath)
                basename = os.path.basename(filename)
                image_count, image_height, image_width = basename.replace("_", "-").split("-")[1:4]
                res += int(image_count)
        return res
    def repeat_paths(self):
        res = []
        for data_source in self.paths:
            res += self.paths[data_source]["all_files"] * self.paths[data_source]["times"] 
        return  res

class TextDecoderAesthetic: 
    def __call__(self , content , 
                use_EN: bool = True , 
                use_CN: bool = False, 
                use_aes_tag : bool = False , 
                use_aes_idx: bool = False ,
                rare_token : str = None,
                index_file : str = None ,
                * args , 
                ** kwargs ):
        if use_EN:
            text_EN = self.get_caption_EN( content , use_aes_tag , index_file)
            if rare_token is not None and text_EN is not None :
                text_EN = text_EN.strip(",").strip(".") + ", " + rare_token
        else:
            text_EN = ""
        if use_CN : 
            text_CN = self.get_caption_CN(content , use_aes_tag , index_file)
            if rare_token is not None and text_EN is not None :
                text_CN = text_CN.strip("，").strip("。") + "，" + rare_token
        else:
            text_CN = ""
        aes_idx = self.get_aes_idx(content)
        return text_EN , text_CN , aes_idx
    @abstractmethod
    def get_caption_EN(self , content , use_aes_tag  , index_file = None ):
        raise NotImplementedError
    @abstractmethod
    def get_caption_CN(self , content , use_aes_tag , index_file = None  ):
        raise NotImplementedError
    @abstractmethod
    def get_aes_idx(self , content):
        raise NotImplementedError
    @staticmethod
    def get_random_aes_idx( ):
        aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '构图' ]
        aes_idx = [0] * len(aes_tag_order)
        for i ,_ in enumerate( aes_tag_order ):
            if random.random() >=0.5:
                aes_idx[i] = 2* i  + 1
            else:
                aes_idx[i] = 2 * i
        return torch.tensor( aes_idx )
    @staticmethod
    def get_empty_aes_idx( ):
        aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '构图' ]
        aes_idx = [0] * len(aes_tag_order)
        for i ,_ in enumerate( aes_tag_order ):
            aes_idx[i] = 2 * i
        return torch.tensor( aes_idx )

class TextDecoderAesthetic_Human(TextDecoderAesthetic):
    def get_caption_EN(self , content , use_aes_tag , index_file = None ):
        if content.get(  "source"  , "" ) == "mj": 
            # print("this is mj data")
            return None
        all_captions = content.get("all_captions", None)
        if all_captions is None:
            return None 
        possible_keys = [
            'blip2_t5', 
            'blip2_opt',
            'Vicuna_LLaVA_short_tags_summary0', 
            'Vicuna_LLaVA_short_tags_summary1', 
            'Vicuna_LLaVA_short_tags_summary2', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 3 words: ', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 5 words: ', 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 8 words: '
        ]
        possible_texts = []
        for pk in possible_keys:
            cur_text = all_captions.get(pk, "")
            if len(cur_text) > 0:
                possible_texts.append(cur_text)
        
        if len(possible_texts) > 0:
            text = random.choice(possible_texts) 
        else:
            return None

        text = text.strip("'").strip('"').strip("., ")

        race = content.get( "race" , "")
        if race != "" and race != "unknown" and random.random()<0.5:
            text = race + ", " + text
        if use_aes_tag : 
            text = self._add_detail_aes_tags(text , content , cn = False )
        return text 
    def get_caption_CN(self , content , use_aes_tag ,index_file = None):
        if content.get(  "source"  , "" ) == "mj": 
            # print("this is mj data")
            return None
        all_captions = content.get("all_captions_cn", None)
        if all_captions is None:
            return None 
        possible_keys = [
            'blip2_opt_cn', 
            'blip2_t5_cn',
            "LLaVA_short_tags_cn"  , 
            "Vicuna_LLaVA_short_tags_summary0_cn" , 
            'Vicuna_LLaVA_short_tags_summary1_cn' , 
            'Vicuna_LLaVA_short_tags_summary2_cn' , 
            "Vicuna_LLaVA_short_tags_Summarize the subject in 3 words: _cn" , 
            'Vicuna_LLaVA_short_tags_Summarize the subject in 5 words: _cn',
            'Vicuna_LLaVA_short_tags_Summarize the subject in 8 words: _cn' , 
        ]

        possible_texts = []
        for pk in possible_keys:
            cur_text = all_captions.get(pk, "")
            if len(cur_text) > 0:
                possible_texts.append(cur_text)
        
        if len(possible_texts) > 0:
            text = random.choice(possible_texts) 
        else:
            return None
        race_mapping = {
            'white' : "白人" , 
            'asian' : "亚洲人" , 
            "black" : "黑人" ,
            'indian' : "印度人" , 
            'west asian' : "东亚人" , 

        }
        race = content.get( "race" , "")
        if race != "" and race != "unknown" and random.random()<0.5:
            text = race_mapping[race] + "，" + text
        if use_aes_tag:
            text = self._add_detail_aes_tags(text , content , cn = True )
        return text 
    def _add_detail_aes_tags(self,text : str ,  item: Dict[str, Any] , cn : bool = False) -> str:
        #aes tag example: ['3.5', '光影', '质感']
        aes_tags = item.get('aes_tags',[])
        aes_map = {
            '不合格': 'normal quality',
            '3.5': 'high quality',
            '4/5': 'best quality',
            '光影': 'chiaroscuro',
            '重心': 'visual focus',
            '质感': 'textured, detailed',
            '色彩': 'aesthetic color',
            '情感': 'emotional',
            '颜值': 'good looking',
            '构图': 'visual arrangement',
        }
        aes_cn_map = {
            '不合格': '标准',
            '3.5': '优质',
            '4/5': '最优',
            '光影': '光影',
            '重心': '重心',
            '质感': '质感',
            '色彩': '色彩',
            '情感': '情感',
            '颜值': '颜值',
            '构图': '构图',
        }
        if cn :
            aes_words = []
            for tag in aes_tags:
                if tag in aes_cn_map:
                    aes_words.append(aes_cn_map[tag])
                else:
                    # print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
                    pass
            if len(aes_words) > 0:
                aes_words = "，".join( aes_words )
                text  = text.strip("，").strip("。")+ "，"  + aes_words
            return text 
        else:
            aes_words = []
            for tag in aes_tags:
                if tag in aes_map:
                    aes_words.append(aes_map[tag])
                else:
                    print(f"didn't find {tag} in aes_map keys:{aes_map.keys()}")
            if len(aes_words) > 0:
                aes_words = ", ".join( aes_words )
                text  = text.strip(",").strip(".")+ ", "  + aes_words
            return text 
    def get_aes_idx(self, content):
        aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '构图' ]
        aes_idx = [0] * len(aes_tag_order)
        aes_tags = content.get('aes_tags',[])
        for i ,aes_tag in enumerate( aes_tag_order ):
            if aes_tag in aes_tags:
                aes_idx[i] = 2* i  + 1 
            else:
                aes_idx[i] = 2 * i
        return torch.tensor( aes_idx )

class TextDecoderAesthetic_Commic(TextDecoderAesthetic):
    def get_caption_CN(self, content, use_aes_tag  , index_file = None):
        text = content.get("blip2_opt_cn" , None )
        if text is None :
            return text 
        label = self.get_style_tag(content , cn = True )
        aes_tag = self._add_detail_aes_tags(content , cn = True )
        if label != "":
            text = text.strip("，").strip("。")
            if random.random() > 0.5 :
                text = f"{label}，{text}。"
            else:
                text = f"{text}，{label}。"
        if aes_tag != '' and use_aes_tag : 
            text = text.strip("，").strip("。")
            text = text + "，" + aes_tag
        return text 
    def get_caption_EN(self, content, use_aes_tag , index_file = None):
        text = content.get("blip2_opt" , None )
        if text is None :
            return text 
        label = self.get_style_tag(content , cn = False )
        aes_tag = self._add_detail_aes_tags(content , cn = False )
        if label != "":
            text = text.strip(",").strip(".")
            if random.random() > 0.5 :
                text = f"{label}, {text}."
            else:
                text = f"{text}, {label}."
        if aes_tag != '' and use_aes_tag : 
            text = text.strip(",").strip(".")
            text = text + "," + aes_tag
        return text 
    def get_aes_idx(self, content):
        aes_dict = {
        5: '光影', # 光影
        7: '重心', # 重心
        4: '质感', # 质感-细节材质
        6: '构图', # 构图-视觉排布
        8: '情感', # 情感-情绪表达
        3: '色彩', # 色彩-颜色搭配
        9: 'good looking', # 颜值-好看潮流

        0: '不合格', # < 3.5
        1: '3.5', # 3.5
        2: '4/5' # 4/5
        }
        aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '构图' ]
        aes_idx = [0] * len(aes_tag_order)
        aes_tags = []
        if content.get('fine_grained'):
            aes_index = content.get('fine_grained',[])
            for idx in aes_index:
                aes_tags.append( aes_dict.get(idx, '') )
        for i ,aes_tag in enumerate( aes_tag_order ):
            if aes_tag in aes_tags:
                aes_idx[i] = 2* i  + 1 
            else:
                aes_idx[i] = 2 * i
        return torch.tensor( aes_idx )
    def get_style_tag(self , content , cn):
        idx2tags_v3 = {
                    0:['anime'], # 占位 备用
                    1:['japanese anime, celluloid'], # 赛璐璐
                    2:['japanese anime, semi-thick painting'], # 半厚涂
                    3:['japanese anime, retro manga'], # 复古昭和
                    4:['japanese anime, woodblock prints'], # 日式版画
                    5:['japanese anime, chibi'], # Q版人物
                    6:['japanese anime, acrylic painting'], # 厚涂
                    7:['anime, fantasy, realistic'], # 梦幻超写实
                    8:['american comics, 2D'], # 2D美漫
                    9:['american comics, acrylic painting'], # 厚涂美漫
                    10:['american cartoon'], # 美式怪诞
                    11:['american comics, retro comics'], # 复古美漫
                    12:['pixar, 3D'], #  卡通3D
                    13:['chinese painting, watercolor painting'], #  水彩/墨
                    14:['chinese anime, ancient chinese'], # 古风仙侠
                    15:['anime, webtoon'], # 现代都市
                    16:['anime'], # 动漫其他
                    17:[''] # 非动漫
                        }
        idx2tags_v3_cn = {
                    0:['日漫'], # 占位 备用
                    1:['日漫， 赛璐璐'], # 赛璐璐
                    2:['日漫， 半厚涂'], # 半厚涂
                    3:['日漫， 复古昭和'], # 复古昭和
                    4:['日漫， 日式版画'], # 日式版画
                    5:['日漫， Q版人物'], # Q版人物
                    6:['日漫， 厚涂'], # 厚涂
                    7:['日漫， 梦幻超写实'], # 梦幻超写实
                    8:['2D美漫'], # 2D美漫
                    9:['厚涂美漫'], # 厚涂美漫
                    10:['美式怪诞'], # 美式怪诞
                    11:['复古美漫'], # 复古美漫
                    12:['卡通3D'], #  卡通3D
                    13:['水彩， 水墨'], #  水彩/墨
                    14:['古风， 仙侠'], # 古风仙侠
                    15:['漫画， 现代都市'], # 现代都市
                    16:['动漫，其他'], # 动漫其他
                    17:[''] # 非动漫
                        }
        style_score = content.get('comic_score_v4',content.get('comic_score_v3_array', content.get('comic_score_v3',content.get('comic_score_v2',[0]))))
        if len(style_score) == 1 or len(style_score) == 0:
            style_idx = 0
        else:
            style_idx = np.argmax(style_score) + 1
        if style_idx < 17:
            #randomly dropout 
            if random.random() < 0.1 :
                return ""
            if not cn:
                return random.choice(idx2tags_v3[style_idx])
            else:
                return random.choice(idx2tags_v3_cn[style_idx])
        return ''
    def _add_detail_aes_tags(self , item: Dict[str, Any] , cn : bool = False ):
        '''
        光影-明暗对比-chiaroscuro
        重心-视觉重心- visual weight / visual focus
        质感-细节材质- textured / detailed / exquisite / high detail / hyper quality
        构图-视觉排布-composition / visual arrangement
        情感-情绪表达-Emotional
        色彩-颜色搭配-aesthetic color / good color combination / gorgeous color scheme
        颜值-好看潮流- lovely face / good looking
        '''
        aes_dict = {
            5: 'chiaroscuro', # 光影
            7: 'visual focus', # 重心
            4: 'textured, detailed', # 质感-细节材质
            6: 'visual arrangement', # 构图-视觉排布
            8: 'emotional', # 情感-情绪表达
            3: 'aesthetic color', # 色彩-颜色搭配
            9: 'good looking', # 颜值-好看潮流

            0: 'normal quality', # < 3.5
            1: 'high quality', # 3.5
            2: 'best quality' # 4/5
        }
        aes_dict_cn = {
            5: '光影', # 光影
            7: '重心', # 重心
            4: '质感', # 质感-细节材质
            6: '构图', # 构图-视觉排布
            8: '情感', # 情感-情绪表达
            3: '色彩', # 色彩-颜色搭配
            9: '颜值', # 颜值-好看潮流

            0: '标准', # < 3.5
            1: '优质', # 3.5
            2: '最优' # 4/5
        }

        aes_tag = ''
        aes_tag_cn = ''
        if cn :
            if item.get('fine_grained'):
                aes_index = item.get('fine_grained',[])
                for idx in aes_index:
                    # The 'emotional' tag is often associated with portraits, so it is not used here temporarily.
                    if idx == 8 :
                        continue
                    aes_tag_cn += '，' + aes_dict_cn.get(idx, '')
            return aes_tag_cn.strip('，')
        else:
            if item.get('fine_grained'):
                aes_index = item.get('fine_grained',[])
                for idx in aes_index:
                    #The 'emotional' tag is often associated with portraits, so it is not used here temporarily.
                    if idx == 8 :
                        continue
                    aes_tag += ', ' + aes_dict.get(idx, '')
            return aes_tag.strip(',')

class TextDecoderAesthetic_Chinese(TextDecoderAesthetic):
    def get_caption_CN(self, content, use_aes_tag , index_file = None):
        anno_data = content.get("anno_data" , None )
        if anno_data is None :
            return None 
        captions = anno_data.get("captions" , None )
        if captions is None :
            return None 
        text = captions.get("Chi_main_introduction" , None )
        if text is None :
            return text
        _ , shoot_language_cn  = self.decode_lens_language(content)
        if shoot_language_cn :
            text= text.strip("。").strip("，") +  "，" + shoot_language_cn
        return text 

    def get_caption_EN(self, content, use_aes_tag , index_file = None):
        anno_data = content.get("anno_data" , None )
        if anno_data is None :
            return None 
        captions = anno_data.get("captions" , None )
        if captions is None :
            return None 
        text = captions.get("Eng_main_introduction" , None )
        if text is None :
            text = content.get("blip2_opt_text" , None )
        if text is None :
            return text
        shoot_language_en , _  = self.decode_lens_language(content)
        if shoot_language_en :
            text= text.strip(",").strip(".") +  ", " + shoot_language_en
        return text
    def get_aes_idx(self, content):
        return TextDecoderAesthetic.get_empty_aes_idx()
    def decode_lens_language(self , content):
        anno_data=  content.get("anno_data" , None )
        if anno_data is None :
            return None ,None 
        lens_data   =  anno_data.get("lens_language" , None)
        if lens_data is None :
            return None ,None 
        try:
            lens_attr = lens_data["attribute"]["index"]  # (场景类，主体类，无法选择)
            lens_scene = lens_data["scene"]["index"]  # (大远景镜头，远景镜头，全景镜头，中景镜头，近景/特写镜头，大特写镜头，无法选择)
            lens_angle = lens_data["camera_shooting_angle"]["index"]  # (平视镜头，俯视镜头，仰视镜头，鸟瞰镜头，斜角镜头，无法选择)
        except:
            return None ,None 
        if lens_attr == 0:
            lens_attr_en = "scene"
            lens_attr_cn = "场景"
        elif lens_attr == 1:
            lens_attr_en = "object"
            lens_attr_cn = "主体"
        else:
            lens_attr_en = ""
            lens_attr_cn = ""
        if lens_scene == 0:
            lens_scene_en = "extreme long shot"
            lens_scene_cn = "大远景镜头"
        elif lens_scene == 1:
            lens_scene_en = "long shot"
            lens_scene_cn = "远景镜头"
        elif lens_scene == 2:
            lens_scene_en = "panoramic shot"
            lens_scene_cn = "全景镜头"
        elif lens_scene == 3:
            lens_scene_en = "medium shot"
            lens_scene_cn = "中景镜头"
        elif lens_scene == 4:
            lens_scene_en = "close-up shot"
            lens_scene_cn = "近景/特写镜头"
        elif lens_scene == 5:
            lens_scene_en = "extreme close-up shot"
            lens_scene_cn = "大特写镜头"
        else:
            lens_scene_en = ""
            lens_scene_cn = ""
        if lens_angle == 0:
            lens_angle_en = "head-up view"
            lens_angle_cn = "平视视角"
        elif lens_angle == 1:
            lens_angle_en = "overlook view"
            lens_angle_cn = "俯视视角"
        elif lens_angle == 2:
            lens_angle_en = "look up view"
            lens_angle_cn = "仰视视角"
        elif lens_angle == 3:
            lens_angle_en = "bird's eye view"
            lens_angle_cn = "鸟瞰视角"
        elif lens_angle == 4:
            lens_angle_en = "bevel view"
            lens_angle_cn = "斜角视角"
        else:
            lens_angle_en = ""
            lens_angle_cn = ""
        shoot_language_en = lens_scene_en + ", " + lens_angle_en + ", " + lens_attr_en
        shoot_language_en = shoot_language_en.strip().strip(",")
        shoot_language_cn = lens_scene_cn + "，" + lens_angle_cn + "，" + lens_attr_cn 
        shoot_language_cn = shoot_language_cn.strip().strip("，").strip("。")
        return shoot_language_en , shoot_language_cn

class TextDecoderAesthetic_General(TextDecoderAesthetic ):
    def get_caption_EN(self, content, use_aes_tag , index_file = None):
        # for VCG  data 
        if index_file is not None and  "vcg" in  index_file:
            return content.get( "blip2_opt_text"  , content.get( "blip2_opt_text"  , None))
        #for  data collected by human
        if index_file is not None and  "manual_collection_cn" in  index_file:
            return content.get( "blip2_opt_pt2_text" ,  None  )
        
        text = self.get_caption(content , [ "blip2_opt_text" , "blip2_t5_text" ,"llava_short_text" , "blip2_opt_pt2_text" ])
        if text is None :
            return text 
        aes_tag = self.get_aes_tag(content , cn = False)
        if use_aes_tag and aes_tag != "" :
            text = text.strip(",").strip(".") + ", " + aes_tag
        return text
    def get_caption_CN(self, content, use_aes_tag , index_file = None):
        # for VCG data 
        if index_file is not None and  "vcg" in  index_file:
            return  content.get( "img_detail_title" , None )
        #for data collected by human
        if index_file is not None and  "manual_collection_cn" in  index_file:
            if content["translate_dict"] is None :
                return None
            text = eval(content["translate_dict"])["blip2_opt_pt2_cn_text"] 
            if text is None :
                return text
            if random.random()> 0.5 :
                key = content.get("file_text_text" , None )
            else:
                key = content.get("topic_words_text" , None )
            if key is not None :
                if random.random()> 0.5 :
                    text = text.strip("，").strip("。")
                    text = text + "，" + key
                else:
                    key = key.strip("，").strip("。")
                    text = key + "，" + text
            return text
        text = self.get_caption(content , [ "blip2_opt_cn_text" , "blip2_t5_cn_text" ,"llava_short_cn_text" ])
        if text is None :
            return text 
        aes_tag = self.get_aes_tag(content , cn = True)
        if use_aes_tag and aes_tag != "" :
            text = text.strip("，").strip("。") + "，" + aes_tag
        return text
    def get_aes_idx(self, content):
        """
        this fuction will get the aes_tag list.
        """
        aes_tag_map = {
            "quality":['不合格' ,'3.5' ,'4/5' ],
            "chiaroscuro" : ["" , '光影'] , 
            "visual_focus" : ["" , '重心'] , 
            "aesthetic_color": ["" , '色彩'],
            "textured_detailed": ["" , '质感'],
            "visual_arrangement": ["" , '构图'],
            "emotional" : ["" , '情感'],
        }
        aes_tag_order = [ '不合格' , '3.5' ,'4/5' ,'重心' , '光影' , '质感' ,  '色彩' , '构图' ]
        aes_idx = [0] * len(aes_tag_order)
        aes_tags = []

        anno_data = content.get("anno_data" , None )
        if anno_data and isinstance(anno_data , dict):
            for k in anno_data.keys():
                    if k in aes_tag_map:
                        aes_tags.append( aes_tag_map[k][ anno_data[k]  ] )

        for i ,aes_tag in enumerate( aes_tag_order ):
            if aes_tag in aes_tags:
                aes_idx[i] = 2* i  + 1 
            else:
                aes_idx[i] = 2 * i
        return torch.tensor( aes_idx )
    def get_caption(self, item , keys = None , index_file = None  ):
        """
        random choice
        """
        if keys is None :
            caption_keys = ["caption", "text", "blip2_opt", "blip2_t5", "tag", "title", "desc", "tags",
                            "blip2_opt_text", "blip2_t5_text", 'deepdanbooru_caption', "file_text_text",
                            "llava_short_text", "clean_prompts"]
        else:
            caption_keys = keys
        texts = []
        for key in caption_keys:
            texts.append(item.get(key, ""))
        texts = [i for i in texts if i != ""]
        texts = [i for i in texts if i is not None]
        if texts == []:
            return None
        text =  random.choice(texts)
        #remove the space character for tusou data
        if index_file is not None and "tusou" in index_file:
            text = text.replace(" "  , '')
        if isinstance(text , float):
            text = None 
        return text
    def get_aes_tag(self , item , cn ):
        anno_data = item.get("anno_data" , "" )
        if anno_data == "":
            return anno_data
        aes_tag_map = {
            "quality":["normal quality" ,"high quality" ,'best quality' ],
            "chiaroscuro" : ["" , "chiaroscuro"] , 
            "visual_focus" : ["" , 'visual focus'] , 
            "aesthetic_color": ["" , 'aesthetic color'],
            "textured_detailed": ["" , 'textured, detailed'],
            "visual_arrangement": ["" , 'visual arrangement'],
            "emotional" : ["" , 'emotional'],
        }
        aes_tag_cn_map = {
            "quality":["标准" ,"优质" ,'最优' ],
            "chiaroscuro" : ["" , "光影"], 
            "visual_focus" : ["" , '重心'] , 
            "aesthetic_color": ["" , '色彩'],
            "textured_detailed": ["" , '质感'],
            "visual_arrangement": ["" , '构图'],
            "emotional" : ["" , '情感'],
        }
        aes_tag = []
        if cn : 
            for k in anno_data.keys():
                if k in aes_tag_cn_map:
                    # 实验验证  “情感” 标签常常与人像关联，因此此处暂时不用。
                    if k == "emotional" :
                        continue
                    aes_tag.append( aes_tag_cn_map[k][ anno_data[k]  ] )
        else:
            for k in anno_data.keys():
                if k in aes_tag_map:
                    # 实验验证  “情感” 标签常常与人像关联，因此此处暂时不用。
                    if k == "emotional" :
                        continue
                    aes_tag.append( aes_tag_map[k][ anno_data[k]  ] )
        if not cn:
            return ", ".join(aes_tag )
        else:
            return "，".join(aes_tag)

class TextDecoderAesthetic_CommicRaw(TextDecoderAesthetic ):
    def get_caption_CN(self, content, use_aes_tag, index_file=None):
        return ""
    def get_caption_EN(self, content, use_aes_tag, index_file=None):
        return content.get("clip_interrogator_caption_text" , None )
    def get_aes_idx(self, content):
        return TextDecoderAesthetic.get_empty_aes_idx()

class MultiSourceTextDecoder_inpaint(TextDecoder):
    """
    Decode text from json dictionary. Return None if sample cannot be decoded to skip forward.
    """

    def modify_openimage_label(self, label: str) -> str:
        label = label.lower()
        label.replace('&', 'and')
        label = label.replace("(", ",")
        label = label.replace(")", "")
        if random.random() > 0.5:
            if label[0] in ['a', 'e', 'i', 'o', 'u']:
                label = "an " + label
            else:
                label = "a " + label
        return label

    def modify_coco_label(self, label: str) -> str:
        label = label.lower()
        label = label.split('-')[0]
        if label in thing_classes:
            if random.random() > 0.5:
                if label[0] in ['a', 'e', 'i', 'o', 'u']:
                    label = "an " + label
                else:
                    label = "a " + label
        return label

    def modify_label(self, label: str) -> str:
        if label in coco_classes:
            return self.modify_coco_label(label)
        else:
            return self.modify_openimage_label(label)

    def multi_tag_text_making(self, item: Dict[str, Any]) -> Optional[str]:
        """
        customized a text description from multiple sources including labels, blip2-captions, tags
        """
        instance_captions = item.get("instance_captions", [])
        if instance_captions:
            local_captions, labels = instance_captions
            # `background` is the key of stuff
            # bug find bad result if train with `background`
            # so, override `background` with stuff-label for it
            is_stuffs = []
            local_captions_override = []
            for local_caption, label in zip(local_captions, labels):
                is_stuff = local_caption == 'background'
                is_stuffs.append(is_stuff)
                if is_stuff:
                    local_captions_override.append(self.modify_label(label))
                else:
                    local_captions_override.append(local_caption)

            if random.random() > 0.5:
                return (local_captions_override, is_stuffs)
            else:
                return ([self.modify_label(label) for label in labels], is_stuffs)

        text_danbooru = item.get("deepdanbooru_caption", "") if random.randint(0, 1) == 1 else ""

        if 'blip2_opt' in item and 'blip2_t5' in item:
            text = item.get(random.choice(["blip2_opt", "blip2_t5"]), "")
        elif 'blip2_opt_text' in item and 'blip2_t5_text' in item:
            text = item.get(random.choice(["blip2_opt_text", "blip2_t5_text"]), "")
        else:
            text = ""

        if text == "":
            # in case no blip2 caption available, find origianl text descriptors
            text_keys = []
            for key in item.keys():
                for potential_key in ["caption", "title", "text", "desc"]:
                    if potential_key in key:
                        if key != "text_length":  # this if is for coyo dataset
                            text_keys.append(key)
            if len(text_keys) >= 1:
                text = item.get(random.choice(text_keys), "")

        if text == "" or "tag" in item.keys():
            # try if it is unsplash or pixbay
            text_tag = item.get("tag", "")
            if len(text_tag) > 0 and random.randint(0, 1) == 1:
                text += " " + text_tag

        # danbooru tags such as "boy, fantasy, etc". Add with a 50% chance.
        if random.randint(0, 1):
            text_danbooru = item.get("deepdanbooru_caption")
            if text_danbooru is not None and len(text_danbooru):
                # randomly sample tags within danbooru
                text_danbooru = text_danbooru.split(",")
                subset_size = random.randint(0, len(text_danbooru))
                if subset_size:
                    # randomly shuffle tags
                    random.shuffle(text_danbooru)
                    # sample from tags
                    subset = text_danbooru[:subset_size]
                    text = text + ", ".join(subset)

        return text

    def __call__(self, item: Dict[str, Any], index_file: Optional[str] = None, *args, **kwargs) -> Optional[str]:
        if index_file is not None and 'tuchong' in index_file:
            text = self.fileterd_tuchong(item)
        elif index_file is not None and ('laion' in index_file and 'translated' in index_file):
            text = self.laion_3b_translated(item)
        else:
            text = self.multi_tag_text_making(item)
        return text

    def laion_3b_translated(self, item: Dict[str, Any]) -> Optional[str]:
        text = item.get("ENG TEXT", None)
        if text is None:
            # actually should not use this text, this text is not in english
            text = item.get("TEXT", None)
        return text

    def tuchong_split_zh_en(self, input_text: str) -> Tuple[str, str]:
        """
        split text into chinese and english by iterating through each character
        and check they are letters or chinese, TODO: clean it up if necessary
        """
        index = []
        for iid, text in enumerate(input_text):
            if not text.isascii():
                index.append(iid)
        if len(index) > 0:
            zh_text = input_text[index[0]: index[-1]]
            en_text = input_text[:index[0]] + input_text[index[-1] + 1:]
            return (zh_text, en_text) if len(en_text) > 5 else (zh_text, "")
        return ("", input_text)

    def tuchong_text_filtered(self, text_original: str, text: str) -> Tuple[str, str]:
        """
        tuchong text processing, select one from two given resources, favor the first one TODO: clean it up if necessary
        """
        if text_original != "":
            return self.tuchong_split_zh_en(text_original)
        if text != "":
            return self.tuchong_split_zh_en(text)
        return ("", "")

    def fileterd_tuchong(self, item: Dict[str, Any]) -> Optional[str]:
        """
        ugly tuchong text processing, TODO: clean it up if necessary
        """
        title_original = item.get("title_original", "")
        title = item.get("title", "")
        keywords_original = item.get("keywords_original", "")
        keywords = item.get("keywords", "")
        remark = item.get("remark_original", "")
        remark_original = item.get("remark", "")

        res = []
        _, res_en = self.tuchong_text_filtered(title_original, title)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)

        # 0426 temporarily remove data only with keywords
        # _, res_en = self.tuchong_text_filtered(keywords_original, keywords)
        # if res_en != "" and len(res_en) > 3:
        #     res.append(res_en)

        _, res_en = self.tuchong_text_filtered(remark, remark_original)
        if res_en != "" and len(res_en) > 3:
            res.append(res_en)
        if len(res) > 0:
            return random.choice(res)
        return None

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os 
import cv2 
from tqdm import tqdm 
from PIL import Image, ImageDraw, ImageFont


def prompt2image(text, canvas=None, line_length=10, font_size=14, height=512, width=512):
    
    if canvas is None:
        image = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        image = Image.fromarray(canvas)
        
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/mnt/bn/dreamlight-bytenas-vg/wpxiao/projects/ReLight/examples/controlnet/utils/dataset_utils/Arial.ttf', font_size)
    x, y = 10, 10

    text_list = text.split(' ')
    lines = []
    for i in range(0, len(text_list)//line_length+1):
        cur_line = ' '.join(text_list[i*line_length:(i+1)*line_length])
        lines.append(cur_line)

    for line in lines:
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += font.getsize(line)[1]

    return np.array(image)


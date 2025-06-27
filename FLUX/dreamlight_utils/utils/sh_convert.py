import random

import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
from scipy.special import sph_harm
import time
import imageio 
from tqdm import tqdm 
import random
import OpenEXR


def real_sph_harm(l, m, theta, phi):
    """计算实数形式的球谐函数"""
    if m > 0:
        return np.sqrt(2) * (-1) ** m * sph_harm(m, l, phi, theta).real
    elif m == 0:
        return sph_harm(m, l, phi, theta).real
    else:
        return np.sqrt(2) * (-1) ** m * sph_harm(-m, l, phi, theta).imag


def spherical_harmonics_coeffs(image, order=2, num_samples=10000):
    """计算给定图像的球谐系数"""
    height, width, _ = image.shape
    coeffs = np.zeros((3, (order + 1) ** 2), dtype=np.float32)

    # coeff = []
    # values = np.load("sample-coeff.npy")
    for _ in range(num_samples):
        # 在单位球面上均匀采样
        theta = np.arccos(2 * np.random.rand() - 1)  # theta: 0 to pi
        val = np.random.rand()
        # coeff.append(val)
        phi = 2 * np.pi * np.random.rand() # phi: 0 to 2pi

        # 将球面坐标转换为图像坐标
        y = int(theta * height / np.pi)
        x = int(phi * width / (2 * np.pi))

        # 确保坐标在图像范围内
        y = np.clip(y, 0, height - 1)
        x = np.clip(x, 0, width - 1)

        pixel = image[y, x] / 255.0  # normalize pixel
        for l in range(order + 1):
            for m in range(-l, l + 1):
                coef_index = l * l + l + m
                Y = real_sph_harm(l, m, theta, phi)  # 使用实数形式的球谐函数
                for c in range(3):
                    coeffs[c, coef_index] += pixel[c] * Y

    return coeffs * (4 * np.pi) / num_samples

def spherical_harmonics_coeffs_v2(image, order=2, num_samples=10000):
    """计算给定图像的球谐系数"""
    height, width, _ = image.shape
    coeffs = np.zeros((3, (order + 1) ** 2), dtype=np.float32)

    # theta_sample = np.array([np.arccos(2 * np.random.rand() - 1) for _ in range(num_samples)])
    theta_sample = np.arccos(2*np.random.rand(num_samples)-1)
    # phi_sample = np.array([2 * np.pi * np.random.rand() for _ in range(num_samples)])
    phi_sample = 2 * np.pi * np.random.rand(num_samples)
    y_sample = (theta_sample * height / np.pi).astype(int)
    x_sample = (phi_sample * width / (2 * np.pi)).astype(int)
    y_sample = np.clip(y_sample, 0, height - 1)
    x_sample = np.clip(x_sample, 0, width - 1)
    index_sample = y_sample * width + x_sample

    pixel_image = image.reshape([-1, 3])[index_sample]
    pixel_image = pixel_image.astype(np.float32) / 255

    for l in range(order + 1):
        for m in range(-l, l + 1):
            coef_index = l * l + l + m
            Y = real_sph_harm(l, m, theta_sample, phi_sample)  # 使用实数形式的球谐函数
            Y = np.expand_dims(Y, axis=1).repeat(3, axis=1)
            coeffs[:, coef_index] += np.sum(pixel_image * Y, axis=0)

    return coeffs * (4 * np.pi) / num_samples


def generate_spherical_image(coeffs, width, height, order=2):
    """根据球谐系数生成球面图像"""
    result = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            theta = np.pi * y / height
            phi = 2 * np.pi * x / width
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    coef_index = l * l + l + m
                    Y = real_sph_harm(l, m, theta, phi)  # 使用实数形式的球谐函数
                    for c in range(3):
                        result[y, x, c] += coeffs[c, coef_index] * Y
    result = np.maximum(result, 0)
    result = normalize(result)
    return result


def generate_spherical_image_v2(coeffs, width, height, order=2):
    """根据球谐系数生成球面图像"""
    result = np.zeros((height, width, 3), dtype=np.float32)
    theta = np.repeat(np.expand_dims(np.arange(height), axis=1), width, axis=1) * np.pi / height
    phi = np.repeat(np.expand_dims(np.arange(width), axis=0), height, axis=0) * 2 * np.pi / width
    coeffs = coeffs.copy()
    coeffs = np.expand_dims(np.expand_dims(coeffs, axis=0), axis=0).repeat(height, axis=0).repeat(width, axis=1)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            coef_index = l * l + l + m
            Y = real_sph_harm(l, m, theta, phi)  # 使用实数形式的球谐函数
            result += np.expand_dims(Y, axis=2).repeat(3, axis=2) * coeffs[:, :, :, coef_index]
    result = np.maximum(result, 0)
    result = normalize(result)
    return result


def generate_spherical_image_v3(coeffs, width, height, order=2):
    """根据球谐系数生成球面图像"""
    result = np.zeros((height, width, 3), dtype=np.float32)
    theta = np.repeat(np.expand_dims(np.arange(height), axis=1), width, axis=1) * np.pi / height
    phi = np.repeat(np.expand_dims(np.arange(width), axis=0), height, axis=0) * 2 * np.pi / width
    coeffs = coeffs.copy()
    coeffs = np.expand_dims(np.expand_dims(coeffs, axis=0), axis=0).repeat(height, axis=0).repeat(width, axis=1)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            coef_index = l * l + l + m
            Y = real_sph_harm(l, m, theta, phi)  # 使用实数形式的球谐函数
            result += np.expand_dims(Y, axis=2).repeat(3, axis=2) * coeffs[:, :, :, coef_index]
    result = np.maximum(result, 0)
    # result = normalize(result)
    return result

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / max(max_val - min_val, 1e-5)
    scaled_array = normalized_array * 255
    final_array = np.round(scaled_array).astype(np.uint8)
    return final_array


def main(image_path, output_path, pano_size=512):
    image = cv2.imread(image_path)

    image = cv2.resize(image, [pano_size, pano_size])
    coeffs = spherical_harmonics_coeffs(image, order=2)
    panorama = generate_spherical_image(coeffs, image.shape[1], image.shape[0])
    panorama = cv2.resize(panorama, [image.shape[1], image.shape[0]])

    cv2.imwrite(output_path, panorama)


def read_envmap(path):
    if path.endswith(".hdr") or path.endswith(".hdri"):
        # image = imageio.imread(path)

        hdr_image = cv2.imread(path, -1)
        ldr_image = np.clip(hdr_image, 0, 1)

        # # Gamma correction
        gamma = 2.2
        ldr_image = np.power(ldr_image, 1/gamma)

        # Normalize the image to 8-bit range (0-255)
        # ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
        # cv2.imwrite("sample-hdr.png", ldr_image*255.)
        # print(">>>>", hdr_image.min(), "~", hdr_image.max())
        # return hdr_image
        return ldr_image
    elif path.endswith(".exr"):
        hdr_image = OpenEXR.File(path).parts[0].channels
        try:
            hdr_image = hdr_image["RGBA"].pixels[...,:3]
        except:
            hdr_image = hdr_image["RGB"].pixels
        ldr_image = hdr_image[...,::-1]#
        # breakpoint()
        # return ldr_image # early return if already filtered by SH converter
        ldr_image = np.clip(ldr_image, 0, 1)
        
        gamma = 2.2
        ldr_image = np.power(ldr_image, 1/gamma)
        
        # cv2.imwrite("sample-exr.png", np.clip(ldr_image*255., 0., 255.))
        # print("EXR:", hdr_image.shape, hdr_image.min(), "~", hdr_image.max())
        return ldr_image
    else:
        # image = np.array(Image.open(path).convert("RGB"))
        ldr_image = cv2.imread(path)
        ldr_image = ldr_image.astype(np.float32)/255.
        gamma = 2.2
        ldr_image = np.power(ldr_image, 1/gamma)
        return ldr_image


def random_spolight_envmap(height=512, width=512):
    pass


    
import torch.nn.functional as F
import PIL.Image
import numpy as np
import torchvision.transforms.functional as TF
from PIL import ImageOps, ImageFilter, ImageDraw
from numpy import random
from torch import nn
from torchvision.transforms import transforms as T, GaussianBlur
from torchvision.transforms import InterpolationMode
import math
import random
import cv2


def compute_cdf(hist):
    # Calculate the cumulative distribution function from the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())  # Normalize
    return cdf_normalized


def match_histograms(image, ref_hist):
    # Calculate the histogram for the image
    hist_img = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    cv2.normalize(hist_img, hist_img)

    # Compute the CDF for image and reference
    cdf_img = compute_cdf(hist_img)
    cdf_ref = compute_cdf(ref_hist)

    # Create a lookup table
    lookup_table = np.zeros(256)
    for i in range(256):
        diff_cdf = np.abs(cdf_ref - cdf_img[i])
        closest_index = np.argmin(diff_cdf)
        lookup_table[i] = closest_index

    # Apply the mapping to the image
    matched_image = cv2.LUT(image, lookup_table)
    return matched_image


def train_transformation1():
    return T.Compose([T.Resize((128, 128), InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(p=0.25),
                      T.RandomVerticalFlip(p=0.25),
                      T.RandomRotation(degrees=30),
                      T.RandomPerspective(distortion_scale=0.5, p=0.25),
                      T.RandomApply([
                          T.ColorJitter(brightness=0.2, contrast=0.2),
                          T.GaussianBlur(kernel_size=3),
                          T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                          T.ElasticTransform(alpha=(50.0, 250.0), sigma=(5.0, 10.0))
                      ], p=0.25),
                      T.Grayscale(3),
                      T.ToTensor(),
                      T.Normalize((0.5,), (0.5,)),
                      # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                      ])


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, crop_size=(400, 400), resize=(128, 128), nopad=True):

        # if isinstance(crop_size, numbers.Number):
        #     self.size = (int(crop_size), int(crop_size))
        # else:
        #     self.size = crop_size
        self.crop_size = crop_size
        self.resize = resize
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, centroid=None):
        w, h = img.size
        # ASSUME H, W
        th, tw = self.crop_size
        if w == tw and h == th:
            return img

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)).resize(size=self.resize, resample=PIL.Image.LANCZOS)


class SobelFilter():
    def __call__(self, img):
        img_x = img.filter(ImageFilter.FIND_EDGES)
        img_y = img.transpose(
            PIL.Image.FLIP_LEFT_RIGHT).filter(ImageFilter.FIND_EDGES).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return PIL.Image.blend(img_x, img_y, alpha=0.5)


class GaussianNoise(object):
    def __init__(self, std=10):
        self.std = std

    def __call__(self, img):
        # Generate the noise array
        noise = np.random.normal(0, self.std, img.size)

        # Add the noise to the image array
        noisy_image_array = img + noise

        # Clip the pixel values to the valid range of 0-255
        noisy_image_array = np.clip(noisy_image_array, 0, 255)

        # Convert the noisy image array back to an image
        return PIL.Image.fromarray(noisy_image_array.astype(np.uint8))


class CustomRotation:
    def __init__(self, angles=None):
        if angles is None:
            angles = [0, 90, 180, 270]
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle)


class RandomErasingPIL:
    def __init__(self, scale=(0.02, 0.1), ratio=(0.3, 3)):
        """
        scale: A tuple indicating the range of the proportion of the erased area against the entire image area.
        ratio: A tuple specifying the range of the aspect ratio of the erased area.
        """
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        for _ in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)
                img = img.copy()
                ImageDraw.Draw(img).rectangle([x1, y1, x1 + w, y1 + h], fill=random.randint(0, 255))
                return img

        return img


class MatchHistogramsTransform:
    def __init__(self, ref_hist=None):
        if ref_hist is None:
            ref_hist = np.load("../utils/avg_hist.npy")
        self.ref_hist = ref_hist

    def __call__(self, image):
        # Assuming the input image is a PIL image
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Apply the histogram matching
        matched_image = match_histograms(image_np, self.ref_hist)

        # Convert numpy array back to PIL image
        matched_image_pil = PIL.Image.fromarray(matched_image).convert("L")
        return matched_image_pil


def get_finetune_transformation(img_size, mean=0.5, std=0.5):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.LANCZOS),
        CustomRotation(angles=[0, 90, 180, 270]),
        MatchHistogramsTransform(),
        T.RandomApply([T.ColorJitter(0.2, 0.2)], p=0.2),
        T.RandomApply([GaussianBlur(kernel_size=int(5), sigma=(0.25, 0.75))], p=0.2),
        T.RandomApply([SobelFilter()], p=0.2),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize((mean,), (std,))
    ])


def get_base_transform(size):
    return [T.Resize((size, size), InterpolationMode.LANCZOS),
            T.Grayscale(3),
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,))
            ]


def get_nst_transform(size):
    return T.Compose([T.Resize(size=(size, size), interpolation=InterpolationMode.LANCZOS),
                      T.Grayscale(3),
                      T.ToTensor()])


def representation_transform(img):
    transform = T.Compose([T.RandomHorizontalFlip(p=0.5),
                           T.RandomRotation(degrees=45),
                           T.RandomPerspective(distortion_scale=0.5, p=0.5),
                           T.RandomGrayscale(p=0.2),
                           T.GaussianBlur(kernel_size=9),
                           T.Grayscale(1),
                           ])
    return transform(img.copy())

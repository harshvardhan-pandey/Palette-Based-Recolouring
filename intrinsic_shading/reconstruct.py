import argparse
import numpy as np
from PIL import Image
import PIL

import matplotlib.pyplot as plt

from chrislib.data_util import load_image, np_to_pil

parser = argparse.ArgumentParser()
parser.add_argument("--alb", help="path to image to decompose")
parser.add_argument("--shd", help="path to image to decompose")
parser.add_argument("--out", help="path to folder to save output image", default="./")

opt = parser.parse_args()

alb = load_image(opt.alb)
shd = load_image(opt.shd)
shd = np.stack([shd, shd, shd], axis=2)

recolored = (np.clip((alb * shd) ** (1 / 2.2), 0, 1) * 255).astype(np.uint8)
recolored_img = Image.fromarray(recolored, mode="RGB")

recolored_img.save(opt.out + "reconstructed.png")
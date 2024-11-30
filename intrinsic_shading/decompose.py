import argparse
import numpy as np
from PIL import Image
import PIL

import torch
import matplotlib.pyplot as plt

from chrislib.general import uninvert
from chrislib.data_util import load_image

from intrinsic.pipeline import load_models, run_gray_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--img", help="path to image to decompose")
parser.add_argument("--out", help="path to folder to save output image", default="./")

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_models("paper_weights", device=device)
img = load_image(opt.img)
result = run_gray_pipeline(model, img, resize_conf=None, maintain_size=True, device=device)

img = result["image"]
alb = (result["gry_alb"] * 255).astype(np.uint8)
shd = (uninvert(result["gry_shd"])[:, :, None] * 255).astype(np.uint8).squeeze()

alb_img = Image.fromarray(alb, mode="RGB")
shd_img = Image.fromarray(shd, mode="L")

alb_img.save(opt.out + "albedo.png")
shd_img.save(opt.out + "shading.png")
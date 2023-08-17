# -*- coding: utf-8 -*-
"""Cloth segmentation
Takes in an image and returns a masked version
Cloth to be worn must be resized and fit over this mask to get further results

"""

from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from cloths_segmentation.pre_trained_models import create_model

!pip install iglovikov_helper_functions
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
!wget https://habrastorage.org/webt/em/l7/cr/eml7crxnxftrimsmolwjegqcrp4.jpeg > /dev/null
!pip install cloths_segmentation  > /dev/null

model = create_model("Unet_2020-10-30")
model.eval();

def run_mask(image):
  transform = albu.Compose([albu.Normalize(p=1)], p=1)
  padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
  x = transform(image=padded_image)["image"]
  x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
  with torch.no_grad():
    prediction = model(x)[0][0]
  mask = (prediction > 0).cpu().numpy().astype(np.uint8)
  mask = unpad(mask, pads)
  return mask

# image = load_rgb("eml7crxnxftrimsmolwjegqcrp4.jpeg") #sample case
flipkart_random_guy=load_rgb("test.jpeg")  # change input for your desirable case

''' 
test.jpeg is an image file from local computer
shape of the mask will be the same as that of input_file, without the channels
If input_shape is (x,y,z) output_shape would be (x,y)
'''
run_mask(image=flipkart_random_guy)
imshow(mask) 
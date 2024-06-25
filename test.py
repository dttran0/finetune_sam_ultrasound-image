from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from draw_bbox import draw_bounding_box  
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize




folder_path = "/home/peter/zebrafish_echo_original/groundtruth_label/"
files = os.listdir(folder_path)



#get the bounding box annotation
bbox = {}

for f in files:
    curr_path = folder_path + f
    bounding_box = draw_bounding_box(curr_path)  
    bbox[f] = np.array(bounding_box[1])
    
#print(bbox)
#get the ground_truth mask annotation
ground_truth_masks = {}
for f in files:
    curr_path = folder_path + f
    gt_grayscale = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_masks[f] = (gt_grayscale == 0)
    
    
model_type = 'vit_t'
checkpoint = "/home/peter/sam/best_model.pth"
device = 'cuda:0'
device_cpu = 'cpu'

mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
print("success")
predictor = SamPredictor(mobile_sam)
print("success")
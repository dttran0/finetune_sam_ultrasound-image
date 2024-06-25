from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from draw_bbox_binary import draw_bounding_box  
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
import segmentation_models_pytorch as smp



folder_path = "/home/peter/zebrafish_echo_original/training_sam_groundtruth/"
files = os.listdir(folder_path)

#get the bounding box annotation
bbox = {}

for f in files:
    curr_path = folder_path + f
    bounding_box = draw_bounding_box(curr_path)  
    bbox[f] = np.array(bounding_box[1])
      
print(len(bbox))
#get the ground_truth mask annotation
ground_truth_masks = {}
for f in files:
    curr_path = folder_path + f
    gt_grayscale = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    gt_grayscale = gt_grayscale[:320, :480]
    ground_truth_masks[f] = (gt_grayscale == 1)

    

model_type = 'vit_b'
checkpoint = "/home/peter/sam/sam_vit_b_01ec64.pth"
device = 'cuda:0'
device_cpu = 'cpu'

mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
print("success")

# Preprocess the images


transformed_data = defaultdict(dict)
for k in bbox.keys():
  #print(k)
  image = cv2.imread("/home/peter/zebrafish_echo_original/training_sam/"+k)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image[:320, :480]
  transform = ResizeLongestSide(mobile_sam.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
  #need to force back the image to the cpu, because then all images will be in gpu
  input_image = mobile_sam.preprocess(transformed_image).to(device_cpu)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])

  transformed_data[k]['image'] = input_image
  transformed_data[k]['input_size'] = input_size
  transformed_data[k]['original_image_size'] = original_image_size
  
print("success preprocess")

# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-7
wd = 0
optimizer = torch.optim.Adam(mobile_sam.mask_decoder.parameters(), lr=lr, weight_decay=wd) #mobile_sam.mask_decoder.parameters()

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()

#loss_fn = smp.utils.losses.DiceLoss()
keys = list(bbox.keys())

batch_mask = None
batch_gt = None

num_epochs = 8000
losses = []
best_val_loss = float("inf")
for epoch in range(num_epochs):
  epoch_losses = []
  count = 0
  for k in keys:
    count += 1
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    
    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
      image_embedding = mobile_sam.image_encoder(input_image)
      #print(bbox[k])
      prompt_box = bbox[k]
      #print(prompt_box)
      box = transform.apply_boxes(prompt_box, original_image_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
      box_torch = box_torch[None, :]
      #print(box_torch)
      sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
          points=None,
          boxes=box_torch,
          masks=None,
      )
      
    low_res_masks, iou_predictions = mobile_sam.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=mobile_sam.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )

    upscaled_masks = mobile_sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    #print(upscaled_masks.min())
    #print(upscaled_masks.max())
    #binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
    binary_mask = threshold(normalize(upscaled_masks), 0.0, 0)
    #print(binary_mask.mean())
    #print(binary_mask.max())
    
    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
    #print(type(gt_binary_mask))
    #print(gt_mask_resized.min())
    #print(gt_mask_resized.max())
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
    #print(gt_binary_mask.mean())
    #print(gt_binary_mask.max())
    if batch_mask == None and  batch_gt == None:
      batch_mask = binary_mask
      batch_gt = gt_binary_mask
    else:
      batch_mask = torch.cat((batch_mask, binary_mask), dim=0)
      batch_gt = torch.cat((batch_gt, gt_binary_mask), dim=0)
    if batch_mask.shape[0] >= 4 or count == 854: 
      loss = loss_fn(binary_mask, gt_binary_mask)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(loss.item())
      batch_mask = None
      batch_gt = None
      #print(loss)
  losses.append(epoch_losses)
  avg_val_loss = np.mean(losses)
  
   # Save model if validation loss has decreased
  if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(mobile_sam.state_dict(), 'best_model_base_sam_dice.pth')
      print(f"Epoch {epoch+1}: Loss improved, saving model...")
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}")

  # Print epoch summary
  # 




  
  
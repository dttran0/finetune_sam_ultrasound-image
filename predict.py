import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "C:\\Users\\trand\\Zebrafish_research\\SAM\\best_model.pth"
#model_type = "vit_h"

model_type = "vit_t"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
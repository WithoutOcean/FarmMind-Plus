from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image,ImageDraw
from sam2_seg.sam2.build_sam import build_sam2
from PIL import Image
import numpy as np
import cv2
from sam2_seg.sam2.sam2_image_predictor import SAM2ImagePredictor
from Retrieve_data import retrieve_temporal,retrieve_context
import re
checkpoint = "/opt/data/private/FSVLM2.0/module/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
segmentation_model= SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
image=Image.open("/opt/data/private/data_set/fiboa/img/NRW/NRW_N32G091022_7.3_52.23_05_patch_3584_512.tif")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    segmentation_model.set_image(image)
    masks, scores, _ = segmentation_model.predict(
            point_coords=None,
            point_labels=[1],
            box=np.array([0,6,47,40])
        )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    
mask = masks[0]
save_dir="/opt/data/private/FSVLM-A-Vision-Language-Model-for-Remote-Sensing-Farmland-Segmentation-main/dataset/test_reason/origin_predict"
save_path = "{}/{}_mask_.png".format(
    save_dir, "/opt/data/private/data_set/fiboa/img/NRW/NRW_N32G091022_7.3_52.23_05_patch_3584_512.tif".split("/")[-1].split(".tif")[0],
)
image_np=np.array(image)
red_layer = np.zeros_like(image_np, dtype=np.float32)
red_layer[mask] = [255, 0, 0] 
red_layer = cv2.cvtColor(red_layer, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_path, red_layer)
print("{} has been saved.".format(save_path))
print("segment over")

import torch
import os
import numpy as np
from PIL import Image,ImageDraw
import base64
from io import BytesIO
import requests
import json
import re
import os

from FarmSeeker_Retrieve_reason_offline import FarmMind_reason_seg
import yaml
from flask import Flask, request, jsonify
import time
import torch
def encode_image(image_file:str):
    with open(image_file,"rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
def load_config(config_path:str):
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.load(f,Loader=yaml.FullLoader)
def get_torch_dtype(dtype_str):
    
    if dtype_str == "torch.bfloat16":
        return torch.bfloat16
    elif dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported data type: {dtype_str}")
cfg = load_config(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "config.yaml",
    )
)
percept_reason_cfg = cfg["perception_reason_seg"]
percept_reason_cfg['data_type'] = get_torch_dtype(percept_reason_cfg['data_type'])
perpcept_reason_seg_module = FarmMind_reason_seg(**percept_reason_cfg)
app=Flask(__name__)
@app.route("/chat", methods=["POST"])
def chat():
    body=request.get_json()
    overlay_img=body.get("overlay_img")
    image_path=body.get("image_path")
    predict_label=np.array(body.get("labels"))
   
    pred_mask=body.get("pred_mask")
    truth_mask=np.array(body.get("truth_mask"))
    
    overlay_img=Image.open(BytesIO(base64.b64decode(overlay_img)))
   
    question1="Please analyze this agricultural remote sensing image (with green farmland segmentation mask overlay) to identify all regions with potential segmentation ambiguity. Provide the bounding box coordinates in [x_min, y_min, x_max, y_max] format, explain the possible causes of ambiguity for each region, and recommend auxiliary data to help refine the segmentation. Output the thinking process in <think> </think> tags and final answer in <answer> </answer> tags. Output the areas with unclear classification in JSON format, along with the types of images that need to be retrieved for each area.   i.e., <think> thinking process here </think>.<answer> { 'bbox1' : [10,100,200,210], 'label1' : 'Retrieve multi-temporal remote sensing images <tool-1>' } </answer>."
    perecpt_result,percept_text=perpcept_reason_seg_module.perceive(overlay_img,question1)
    pred_mask=Image.open(BytesIO(base64.b64decode(pred_mask)))
    draw=ImageDraw.Draw(pred_mask)
    for i, box in enumerate(perecpt_result["bbox"], start=1):
        draw.rectangle(box,outline="red",width=3)
        
    buffered = BytesIO()
    pred_mask.save(buffered, format="PNG")
    pred_mask = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    last_label=perpcept_reason_seg_module.farmland_reason_seg(image_path=image_path,pred_mask=predict_label,bboxes=perecpt_result["bbox"],retrieve_types=perecpt_result["retrieve"],truth_mask=truth_mask)                                              
    last_label=last_label.astype(int).tolist()
    response={"ambiguity_perception":percept_text,"reasoning_correction":last_label}
    return jsonify(
        {
            "response":response
        }
    )
if __name__=="__main__":
  
    app.run(host="0.0.0.0",port=7899)
    print("input your request")

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image,ImageDraw
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ast
from Retrieve_data_offline import retrieve_temporal,retrieve_context
import re
import base64
import json
from io import BytesIO
# default: Load the model on the available device(s)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Any
import torch
import os
import numpy as np
from PIL import Image,ImageDraw
import base64
import requests
import sys
import save_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sam2_seg.sam2.build_sam import build_sam2
from sam2_seg.sam2.sam2_image_predictor import SAM2ImagePredictor

import re
import torch

def get_month(image_path=None,mon_str=""):
    if mon_str=="" and image_path is not None:
        image_name=image_name = os.path.basename(image_path).split(".tif")[0] if ".tif" in image_path else os.path.basename(image_path).split(".png")[0]
        mon_str=image_name.split("_")[4]
    month_int=int(mon_str)
    month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
    }
    return month_dict[month_int]
def parse_ambiguity_text(text: str) -> Dict[str, List[Any]]:
    answer_pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
    match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        print("❌ Failed to find <answer> tag or content")
        return {"bbox": [], "retrieve": []}
    
    content = match.group(1).strip()
    try:
        data_dict = ast.literal_eval(content)
    except Exception as e:
        print(f"❌ Failed to parse answer content: {e}")
        return {"bbox": [], "retrieve": []}
    
    bboxes = []
    retrieve = []
    index = 1
    while True:
        bbox_key = f"bbox{index}"
        label_key = f"label{index}"
        
        if bbox_key in data_dict and label_key in data_dict:
            try:
                box_coords = list(map(int, data_dict[bbox_key]))  
                suggestion = str(data_dict[label_key]).strip()
                
                bboxes.append(box_coords)
                retrieve.append(suggestion)
            except (ValueError, TypeError) as ex:
                print(f"⚠️ Failed to parse entry {index}")
        else:
            break  
        
        index += 1

    return {"bbox": bboxes, "retrieve": retrieve}
class FarmMind_reason_seg:
    def __init__(self,sam_checkpoint,sam_cfg,database_temporal,database_enlarge,crop_temporal_dir,crop_enlarge_dir,model_path,save_path,threshold,device: str = "cuda",data_type=torch.bfloat16):
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=data_type,device_map=device)
        self.processing_class = AutoProcessor.from_pretrained(model_path)
        self.processor=AutoProcessor.from_pretrained(model_path)
        self.segmentation_model = SAM2ImagePredictor(build_sam2(sam_cfg, sam_checkpoint))
        self.termporal_dir=database_temporal
        self.context_dir=database_enlarge
        self.crop_temporal=crop_temporal_dir
        self.crop_context=crop_enlarge_dir
        self.save_path=save_path
        self.threshold=threshold
        self.model.eval()
    def get_season(self,month_str):
        month_int=int(month_str)
        month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}
        return month_dict[month_int]
    def perceive(self, image_label: Any, question: str) -> Dict[str, List[Any]]:
        if not image_label:
             system_prompt="Now you are an intelligent remote sensing interpretation expert, and you need to perform a task of agricultural remote sensing image segmentation. You need to follow the following process steps to complete this task: 1. You need to first call the basic perception segmentation tool to perform basic segmentation on user input; 2. Analyze and determine whether there is any segmentation ambiguity in the basic results, and call the clipping tool to obtain auxiliary data based on the analysis. If there is segmentation ambiguity, proceed to the next step; otherwise, return the result directly; 3. Combined with auxiliary data analysis, determine whether the ambiguous area contains cultivated land and use correction tools to help correct the segmentation results."
             messages = [
                        { 
                        "role": "system",
                        "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                              
                                {"type": "text", "text": question+"Please provide your planning steps."},
                            ],
                        }
                        ]
        else:
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_label}, 
                                {"type": "text", "text": question},
                            ],
                        }
                         ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024,do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not image_label:
            return output_text[0]
        result = parse_ambiguity_text(output_text[0])
       
        return result,output_text[0]
    def qwen_chat(self,question,multi_image):
       
        content=[]
        for img in multi_image:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages = [
        {
            "role": "user",
            "content": content,
        }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024,do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
       
        return output_text[0]
    def sam2_segment(self,image,bbox,answer, last_label,one_bbox,truth_mask,threshold=0.3):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.segmentation_model.set_image(image)
            masks, scores, _ = self.segmentation_model.predict(
                    point_coords=None,
                    point_labels=[1],
                    box=np.array(bbox)
                )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
        mask=masks[0]
        farmland_pixel_num=truth_mask[one_bbox[1]:one_bbox[3],one_bbox[0]:one_bbox[2]].sum()
        ratio_farmland=farmland_pixel_num/((one_bbox[2]-one_bbox[0])*(one_bbox[3]-one_bbox[1]))
        judge_answer="right"
        truth_answer=""
        if ratio_farmland>threshold:
            truth_answer="is farmland"
        else:
            truth_answer="no farmland"
        if "no farmland" in answer:
            last_label[one_bbox[1]:one_bbox[3],one_bbox[0]:one_bbox[2]]=last_label[one_bbox[1]:one_bbox[3],one_bbox[0]:one_bbox[2]]-mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] 
        else: 
            last_label[one_bbox[1]:one_bbox[3],one_bbox[0]:one_bbox[2]]=last_label[one_bbox[1]:one_bbox[3],one_bbox[0]:one_bbox[2]]+mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        if truth_answer not in answer:
            judge_answer="wrong"
        return last_label,judge_answer
    
    def farmland_reason_seg(self,image_path,pred_mask,bboxes,retrieve_types,truth_mask,max_retries=3):
        last_label=pred_mask.copy()
        image_ori_month=get_month(image_path)
       
        image_ori=Image.open(image_path)
        for one_bbox,reg_type in zip(bboxes,retrieve_types):
            bbox=[]
            new_image=image_ori.copy()
            new_seg_image=image_ori.copy()
            draw = ImageDraw.Draw(new_image)
            draw.rectangle([one_bbox[0], one_bbox[1], one_bbox[2] ,one_bbox[3] ], outline="red", width=3)
           
            
            if "<tool-1>" in reg_type:
                temporal_image,temporal_ori,months=retrieve_temporal(image_path,self.termporal_dir,self.crop_temporal,one_bbox)
               
                bbox=one_bbox
                multi_image=[new_image,temporal_image[0]]
                temporal_mon=self.get_season(months[0])
                question=(f"These two remote sensing images were taken in {image_ori_month} and {temporal_mon} respectively, and the area to be determined is marked at the red bounding box<box>{one_bbox}</box>. As a remote sensing interpretation expert, please compare and observe the images from different periods in {image_ori_month} and {temporal_mon} to determine whether the red bounding box area is farmland."
                            f"You can refer to the following aspects for analysis, but please make sure to make comprehensive inferences based on the real situation of the image, and do not list them rigidly: 1 Phenology and color tone evolution, 2 Texture and detail features, 3 Space Geometry and Contour, 4 Environmental context." 
                            f"Please output your thought process in the<think>and</think>tags, and output the final judgment result in the<answer>and</answer>tags. If you determine it is farmland, in the <answer></answer> tag, simply output 'there is farmland within the red bounding box'. If it is not farmland, in the <answer></answer> tag, simply output 'there is no farmland within the red bounding box'.")      
        
               
            else:
                context_image,context_ori,new_bbox=retrieve_context(image_path,self.context_dir,self.crop_context,one_bbox)
                bbox=new_bbox
                multi_image=context_image
                question= (f"This is a remote sensing image with a red bounding box in the<box>{new_bbox}</box>. As a remote sensing interpretation expert, please determine whether the red bounding box area is farmland."
                            f"You can refer to the following aspects for analysis, but please make sure to make comprehensive inferences based on the real situation of the image, and do not list them rigidly: 1 Phenology and color tone evolution, 2 Texture and detail features, 3 Space Geometry and Contour, 4 Environmental context." 
                            f"Please output your thought process in the<think>and</think>tags, and output the final judgment result in the<answer>and</answer>tags. If you determine it is farmland, in the <answer></answer> tag, simply output 'there is farmland within the red bounding box'. If it is not farmland, in the <answer></answer> tag, simply output 'there is no farmland within the red bounding box'.")
            
            
            for _ in range(1,max_retries+1):
                success = False
                response = self.qwen_chat(question, multi_image)
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not answer_match:
                    continue
                last_label,judge_answer=self.sam2_segment(new_seg_image,bbox,answer_match.group(1), last_label,one_bbox,truth_mask,self.threshold)
                success = True
                break
            if not success:
                continue
           
        
        last_label=np.array(last_label)
        last_label[last_label>1]=1
        last_label[last_label<0]=0
        correct_mask=torch.tensor(last_label, dtype=torch.int32).unsqueeze(0)
        save_data.save_mask(correct_mask,image_path,self.save_path,np.array(image_ori))
        return last_label

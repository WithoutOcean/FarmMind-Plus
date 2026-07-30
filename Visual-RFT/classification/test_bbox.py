import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

from transformers.generation import GenerationConfig

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)
import json

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import cv2
# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()




# model path and model base
model_path ="/g0001sr/why/Visual-RFT-origin/out_model" #"/opt/data/private/Visual-RFT-origin/Qwen2.5VL-7B-RF"       

ori_processor_path="/g0001sr/why/Qwen2.5-VL-7B"
data_path="/g0001sr/why/farmland_merge_bbox"
save_dir="/g0001sr/why/Visual-RFT-origin/classification/farmland_merge_bbox"
def run(rank, world_size):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(ori_processor_path) 

    model = model.to(torch.device(rank))
    model = model.eval()
    rank = rank
    world_size = world_size
    with open(data_path,"r",encoding="utf-8") as f :
        data_json=json.load(f)
    error_count = 0
    right_count = 0
    data=[] 
    for sample in data_json[-30:-10]:
        image_path=sample["image_path"]
       
        query="<image>\n"+sample["problem"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024,use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        image=cv2.imread(image_path)
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', response)
            student_answer = content_match.group(1).strip() if content_match else response.strip()
            # student_answer = '<answer>'+student_answer+'</answer>'
          
            
            # fix format error
            student_answer = student_answer.replace("[{",'{')  
            student_answer = student_answer.replace("}]",'}')  
            student_answer = student_answer.replace("\n",'') 
            student_answer =  student_answer.replace("'", '"')
            region_json = json.loads(student_answer)
            for name, points in region_json.items():
                 cv2.rectangle(image,(points[0],points[1]),(points[2],points[2]),(0,255,0),2)
            image_name=image_path.split("/")[-1]
            image_path=os.path.join(save_dir,image_name)
            cv2.imwrite(image_path, image)   
        except:
            image_name=image_path.split("/")[-1]
            image_path=os.path.join(save_dir,image_name)
            cv2.imwrite(image_path, image)       
        
    return [error_count, right_count]

def main():
    multiprocess = torch.cuda.device_count() >= 1
    mp.set_start_method('spawn')
    run(rank=0,world_size=1)
    # if multiprocess:
    #     logger.info('started generation')
    #     n_gpus = torch.cuda.device_count()
    #     world_size = n_gpus
    #     with Pool(world_size) as pool:
    #         func = functools.partial(run, world_size=world_size)
    #         result_lists = pool.map(func, range(world_size))

    #     global_count_error = 0
    #     global_count_right = 0
    #     global_results = []
    #     for i in range(world_size):
    #         global_count_error += int(result_lists[i][0])
    #         global_count_right = global_count_right + result_lists[i][1]
            
    #     logger.info('Error number: ' + str(global_count_error))  
    #     logger.info('Total Right Number: ' + str(global_count_right))
    # else:
    #     logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
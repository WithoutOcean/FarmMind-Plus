

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import ast
from functools import partial
import torch
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
import numpy as np
from scipy.optimize import linear_sum_assignment
from math_verify import parse, verify
import sys
sys.path.append("/g0001sr/why/Visual-RFT-origin/src/virft/src")
from open_r1.trainer import Qwen25VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import json
from dataclasses import dataclass, field
from coor_utils import generalized_box_iou,accum_probs,box_iou
import argparse
import cv2
import torch.distributed as dist

from difflib import SequenceMatcher

import requests
def test(questions,device_idx):
    urls=["http://127.0.0.1:7891/chat","http://127.0.0.1:7892/chat"]
    url=urls[device_idx]
    try:
        response=requests.post(url, json={"question":questions}).json()
    except:
        print("no response")
        
    text_list =response["response"]
    number_float=[]
    for text in text_list:
        text=text.split("</think>")
        pattern = r'\d+'
        matches = re.findall(pattern, text[-1])
        
        if len(matches)>0:
            
            number_str = matches[0]
         
            if float(number_str)>1.0:
                number_float.append(1.0)
            else:
                number_float.append(float(number_str))
        else:
            print("No number found after </think>")
            number_float.append(0.0)
        
    return number_float

def parse_args(args):
    parser = argparse.ArgumentParser(description="Qwen2.5VL-7B-RL")
    parser.add_argument("--base_model", default="",help="The model weights for cold start")
    parser.add_argument("--eval_model", default="",help="Teacher reward model")
    parser.add_argument("--data_path", default="",help="Data for RL")
    parser.add_argument("--epochs", default=1)

    parser.add_argument("--KL_penaly", default=0.05)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--gradient_accumulation_steps", default=1)
    parser.add_argument("--learning_rate", default=5e-6)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--num_generations", default=6)
    parser.add_argument("--temperature", default=1)
    parser.add_argument("--top_p", default=0.9)
    parser.add_argument("--top_k", default=None)
    parser.add_argument("--use_peft", default=True)

    parser.add_argument("--lora_r", default=16)
    parser.add_argument("--lora_alpha", default=32)
    parser.add_argument("--lora_dropout", default=0.05)
    parser.add_argument("--lora_target_modules", default=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"])#
    parser.add_argument("--max_prompt_length", default=2048)
    parser.add_argument("--max_completion_length", default=1024)
 
    parser.add_argument("--model_save_path", default="")
   
    return parser.parse_args(args)
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    eval_match: Optional[bool] = field(default=False)
    adaptive: Optional[bool] = field(default=False)

def message_process(message,processor):
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
    return inputs

def is_valid_bbox_label_format(s):
    try: 
        data = ast.literal_eval(s)
        if not isinstance(data, dict):
            return False 
        if len(data) == 0:
            return False
        bbox_items = {}
        label_items = {}
        for key, value in data.items():            
            bbox_match = re.fullmatch(r'bbox(\d+)', key)
            if bbox_match:
                idx = bbox_match.group(1)               
                if not (isinstance(value, (list, tuple)) and len(value) == 4):
                    return False
                if not all(isinstance(v, (int, float)) for v in value):
                    return False
                bbox_items[idx] = value
                continue

            
            label_match = re.fullmatch(r'label(\d+)', key)
            if label_match:
                idx = label_match.group(1)     
                if not isinstance(value, str):
                    return False
                label_items[idx] = value
                continue
            return False

        bbox_keys = set(bbox_items.keys())
        label_keys = set(label_items.keys())
        if bbox_keys != label_keys:
            return False
        return True  
    except (SyntaxError, ValueError) as e:
        return False
    except Exception:
        return False
def format_reward(completions,solution,**kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.0 if match else -1.0 for match in matches]

def think_accuracy(completions,solution,device_indx,**kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    all_question=[]
    
    for content, sol in zip(contents,solution):
        # if sol not in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
        #     rewards.append(0.0)
        #     continue
        try:
            content_match = re.search(r'<think>.*?</think>', content, re.DOTALL)
            student_think = content_match.group(0).strip() if content_match else content.strip()    
            content_answer = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_answer.group(1).strip() if content_answer else content.strip()
            question=kwargs["problem"][0]
            model_reward_tempt="'{Model_thinking}'is the thinking content output by the model for the problem '{Question}', and '{Model_Answer}' is the answer output by the model. Firstly, check whether the model is thinking according to the requirements of the problem. Secondly, check whether the thinking content is consistent with the output answer. Finally, check whether the thinking content of the model is coherent in word order and logic. If there are no issues with the above three requirements, then it is judged as 1. If any item does not meet the requirements, it will be judged as 0. Only output judgment, do not output any other content"
            all_question.append(model_reward_tempt.format(Question=question,Model_thinking=student_think.lower(),Model_Answer=student_answer.lower()))
            rewards.append(100)
        except Exception:
            rewards.append(0.0)
    if len(all_question)>0:
        new_rewards=[]
        response=test(questions=all_question,device_idx=device_indx)
        idx=0
        for num in rewards:
            if num==100:
                if idx<len(response):
                    new_rewards.append(response[idx])
                    idx+=1
                else:
                   raise ValueError("List b does not have enough elements to replace all 100s in list a.")
            else:
                new_rewards.append(num)
    else:
        new_rewards=rewards
    return new_rewards

def answer_accuracy(completions,solution,**kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards=[]
    for content,sol in zip(contents,solution):
        if sol in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
            try:
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                if student_answer==sol:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def json_format_reward(completions,solution,**kwargs):
    contents=[completion[0]["content"] for completion in completions]
    rewards=[]
    for content,sol in zip(contents,solution):
        if sol in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
            rewards.append(0.0)
            continue
        try:
            content_answer = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_answer.group(1).strip() if content_answer else content.strip()
            match_reward=is_valid_bbox_label_format(student_answer)
            if match_reward:
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        except Exception:
            rewards.append(-1.0)
    return rewards

def parse_json(json_output):
    """
    parse output
    """
    try:
        lines = json_output.strip().splitlines()
        
        # remove Markdown JSON tag
        if lines[0].startswith("```json"):
            json_output = "\n".join(lines[1:])  # remove "```json"
        
        if "```" in json_output:
            json_output = json_output.split("```")[0]  # remove ```
        
        return json.loads(json_output)  # parse JSON
    except Exception:
        return None  # if fail, return None

def extract_bbox_label(text):
    """
    extract  {"bbox_2d": [...], "label": "..."} by re
    """
    try:
        pattern = re.compile(r'\{"bbox_2d": \[(\d{1,3}|1000), (\d{1,3}|1000), (\d{1,3}|1000), (\d{1,3}|1000)\], "label": "([^"]+)"\}')
        matches = pattern.findall(text)

        extracted_data = []
        for match in matches:
            x1, y1, x2, y2, label = match
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            # filter negative bbox（check x1 < x2, y1 < y2）
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                extracted_data.append({"bbox": bbox, "category_name": label})

        return extracted_data
    except Exception:
        return []  # return blank

def extract(completion):
    """
    Extract bbox_2d and lable, and return blank list if failed
    """
    output = []
    try:
        extracted = parse_json(completion)
        if isinstance(extracted, list):  # ensure list output
            for item in extracted:
                try:
                    category_name = item.get("label")
                    bbox = item.get("bbox_2d")

                    # ensure bbox format
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox = [int(coord) for coord in bbox]  # ensure int
                        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:  # filter negative bbox
                            output.append({"bbox": bbox, "category_name": category_name})
                except Exception:
                    continue  # ignore single error and continue
    except Exception:
        pass  # if failed, turn to re match

    if not output:  # if failed, turn to re match
        try:
            output = extract_bbox_label(completion)
        except Exception:
            output = []  

    return output  


def BoxPriorMatcher(outputs, targets):
    bs, num_queries = outputs["pred_boxes"].shape[:2]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (bs * Nq, 4)
    tgt_bbox = torch.cat([v["boxes"] for v in targets])  # (total_Ngt, 4)
    giou_matrix = generalized_box_iou(out_bbox, tgt_bbox)  # (M, N)
    cost_giou = -giou_matrix  
    C = cost_giou.view(bs, num_queries, -1).cpu()  # (bs, Nq, sum(N_gt))
    sizes = [len(v["boxes"]) for v in targets]      
    indices = []
    start = 0
    for i, n_gt in enumerate(sizes):
        end = start + n_gt
        cost_matrix_i = C[i, :, start:end]  # (Nq, n_gt)
        row_idx, col_idx = linear_sum_assignment(cost_matrix_i)
        indices.append((row_idx, col_idx))
        start = end
    results = []
    start = 0
    for i, n_gt in enumerate(sizes):
        end = start + n_gt
        pred_indices, gt_indices = indices[i]
        if len(pred_indices) > 0 and len(gt_indices) > 0: 
            pred_tensor = torch.tensor(pred_indices)
            gt_tensor = torch.tensor(gt_indices)
            gious = giou_matrix[pred_tensor, gt_tensor + start]
            results.append([torch.tensor(pred_indices), torch.tensor(gt_indices), gious])
        else:
            results.append([torch.tensor([]), torch.tensor([]), torch.tensor([])])
        start = end
    return results

def fuzzy_command_match(cmd1: str, cmd2: str) -> bool:
    if not cmd1 or not cmd2:
        return False   
    cmd1 = cmd1.lower().strip()
    cmd2 =cmd2.lower().strip() 
    cmd1 = re.sub(r"\s+", " ", cmd1)
    cmd2 = re.sub(r"\s+", " ", cmd2)
    return cmd1 == cmd2

def modify_list(lst, minimal=0.5, maximum=0.75):
    """
    Change to adapative, when model can well perform than one threshold, change it to a higher value
    """
    return [0 if x < minimal else 1 if x > maximum else x for x in lst]

def recall_reward(completions, solution, step, **kwargs):
    if step is not None:
        modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
        min_iou = 0.5 if step < 1500 else 0.75
    else:
        modify_list_ = partial(modify_list, minimal=0.0, maximum=1.0)
        min_iou = 0.5
    rewards = []
    for completion, sol in zip(completions, solution):
        if sol in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
            rewards.append(0.0)
            continue
        solution_item=ast.literal_eval(sol)    
        raw_content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_content), re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                pred_dict = eval(answer_content)               
            else:
                pred_dict = eval(str(raw_content).strip())      
            dt_bboxes = []
            dt_retrieves = []
            i = 1
            while f"bbox{i}" in pred_dict:
                try:
                    bbox = pred_dict[f"bbox{i}"]
                    label = pred_dict.get(f"label{i}", "").strip()
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        dt_bboxes.append(list(map(float, bbox)))
                        dt_retrieves.append(label)
                    i += 1
                except:
                    i+=1
            gt_bboxes = []
            gt_retrieves = []
            for item in solution_item:
                bbox = item["bbox"]
                retrieve = item.get("retrieve", "").strip()
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    gt_bboxes.append(list(map(float, bbox)))
                    gt_retrieves.append(retrieve)
            num_instance = len(gt_bboxes)  

            if not dt_bboxes:
                recall = 0.0 if num_instance > 0 else 1.0
                rewards.append(modify_list_([recall])[0])
                continue

            dt_bboxes_tensor = torch.tensor(dt_bboxes, dtype=torch.float32).unsqueeze(0)  # (1, Np, 4)
            gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)               # (Ng, 4)
            dt_logits = torch.zeros((1, dt_bboxes_tensor.size(1), 2))
            dt_dict = {"pred_boxes": dt_bboxes_tensor, "pred_logits": dt_logits}
            gt_dict = {"boxes": gt_bboxes_tensor, "labels": torch.zeros(len(gt_bboxes))}

            try:
                match_result = BoxPriorMatcher(dt_dict, [gt_dict])
                matched_gt_indices = set()  
                valid_match_count = 0       
                for match_item in match_result:
                    if len(match_item) >= 3 and len(match_item[0]) > 0:
                        pred_indices, gt_indices, giou_scores = match_item
                        for i in range(len(pred_indices)):
                            if i < len(giou_scores):
                                giou_score = giou_scores[i].item()
                                gt_idx = gt_indices[i].item()
                                if giou_score >= min_iou:
                                    matched_gt_indices.add(gt_idx)  
                                    valid_match_count += 1
                tp = len(matched_gt_indices)   
            except Exception as e:
                print(f"pred_bbox:{dt_bboxes_tensor}")
                tp = 0
            if num_instance > 0:
                recall = tp / num_instance  
                recall = modify_list_([recall])[0]
                rewards.append(recall)
            else:
                rewards.append(0.0)
        except Exception as e:
            rewards.append(0.0) 
    return rewards

def classification_accuracy_reward(completions, solution, step=None, **kwargs):
    if step is not None:
        if step < 1500:
            modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
            min_iou = 0.5
        else:
            modify_list_ = partial(modify_list, minimal=0.75, maximum=0.9)
            min_iou = 0.75
    else:
        modify_list_ = partial(modify_list, minimal=0.0, maximum=1.0)
        min_iou = 0.5
    rewards = []
    for completion, sol in zip(completions, solution):
        if sol in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
            rewards.append(0.0)
            continue
        try:
            solution_item = ast.literal_eval(sol)
        except:
            rewards.append(0.0)
            continue
        raw_content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_content), re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                pred_dict = eval(answer_content)
            else:
                pred_dict = eval(str(raw_content).strip())
            dt_bboxes = []
            dt_retrieves = []
            i = 1
            while f"bbox{i}" in pred_dict and f"label{i}" in pred_dict:
                try:
                    bbox = pred_dict.get(f"bbox{i}")
                    label = pred_dict.get(f"label{i}", "").strip()
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        dt_bboxes.append(list(map(float, bbox)))
                        dt_retrieves.append(label)
                    i += 1
                except:
                    i+=1
            gt_bboxes = []
            gt_retrieves = []
            for item in solution_item:
                bbox = item.get("bbox")
                retrieve = item.get("retrieve", "").strip()
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    gt_bboxes.append(list(map(float, bbox)))
                    gt_retrieves.append(retrieve)
            if not gt_bboxes or not dt_bboxes:
                rewards.append(0.0)
                continue
            dt_bboxes_tensor = torch.tensor(dt_bboxes, dtype=torch.float32).unsqueeze(0)  # (1, Np, 4)
            gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)               # (Ng, 4)
            dt_logits = torch.zeros((1, dt_bboxes_tensor.size(1), 2))
            dt_dict = {"pred_boxes": dt_bboxes_tensor, "pred_logits": dt_logits}
            gt_dict = {"boxes": gt_bboxes_tensor, "labels": torch.zeros(len(gt_bboxes))}
            try:
                match_result = BoxPriorMatcher(dt_dict, [gt_dict])
                matched_gt_indices = set()
            except Exception as e:
                match_result = []
            correct_label_matches = 0
            total_valid_matches = 0  
            try:
                if len(match_result) > 0:
                    for match_item in match_result:
                        if len(match_item) >= 3:  # (pred_indices, gt_indices, giou_scores)
                            pred_indices, gt_indices, giou_scores = match_item
                            for i in range(len(pred_indices)):
                                try:
                                    pred_idx = pred_indices[i].item()
                                    gt_idx = gt_indices[i].item()
                                    if pred_idx >= len(dt_retrieves) or gt_idx >= len(gt_retrieves):
                                        continue
                                    if i < len(giou_scores):
                                        giou_score = giou_scores[i].item()          
                                        if giou_score >= min_iou and gt_idx not in  matched_gt_indices:  
                                            matched_gt_indices.add(gt_idx)
                                            total_valid_matches += 1
                                            
                                            pred_label = dt_retrieves[pred_idx]
                                            gt_label = gt_retrieves[gt_idx]

                                            if fuzzy_command_match(pred_label, gt_label):
                                                correct_label_matches += 1
                                except Exception as e:
                                    continue
            except Exception as e:
                print(f"{e}")
            if total_valid_matches > 0: 
                label_accuracy = correct_label_matches / total_valid_matches
            else:
                label_accuracy = 0.0
            final_reward = modify_list_([label_accuracy])[0]
            rewards.append(final_reward)
        except Exception as e:
            rewards.append(0.0)          
    return rewards
def spatial_precision_reward(completions, solution, step=None, **kwargs): 
   
    if step is not None:
        if step < 1500:
            modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
            min_iou = 0.5
        else:
            modify_list_ = partial(modify_list, minimal=0.75, maximum=0.9)
            min_iou = 0.75
    else:
        modify_list_ = partial(modify_list, minimal=0.0, maximum=1.0)
        min_iou = 0.5

    rewards = []

    for completion, sol in zip(completions, solution):
        
        if sol in ["there is farmland within the red bounding box","there is no farmland within the red bounding box"]:
            rewards.append(0.0)
            continue
        try:
            solution_item = ast.literal_eval(sol)
        except:
            rewards.append(0.0)
            continue
        raw_content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_content), re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                pred_dict = eval(answer_content)
            else:
                pred_dict = eval(str(raw_content).strip())
            dt_bboxes = []
            i = 1
            while f"bbox{i}" in pred_dict:
                try:
                    bbox = pred_dict.get(f"bbox{i}")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        dt_bboxes.append(list(map(float, bbox)))
                    i += 1
                except:
                    i+=1
            gt_bboxes = []
            for item in solution_item:
                bbox = item.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    gt_bboxes.append(list(map(float, bbox)))
            if not dt_bboxes:
                rewards.append(1.0 if not gt_bboxes else 0.0)
                continue
            if not gt_bboxes:
                rewards.append(0.0)
                continue
            dt_bboxes_tensor = torch.tensor(dt_bboxes, dtype=torch.float32).unsqueeze(0)
            gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)
            dt_dict = {"pred_boxes": dt_bboxes_tensor, "pred_logits": torch.zeros((1, dt_bboxes_tensor.size(1), 2))}
            gt_dict = {"boxes": gt_bboxes_tensor, "labels": torch.zeros(len(gt_bboxes))}
            try:
                match_result = BoxPriorMatcher(dt_dict, [gt_dict])
                matched_gt_indices = set()                 
                if len(match_result) > 0:
                    for match_item in match_result:
                        if len(match_item) >= 3:
                            pred_indices, gt_indices, giou_scores = match_item
                            for i in range(len(gt_indices)):
                                if i < len(giou_scores):
                                    giou_score = giou_scores[i].item()
                                    gt_idx= pred_indices[i].item()
                                    if giou_score >= min_iou:
                                        matched_gt_indices.add(gt_idx)               
                total_tp = len(matched_gt_indices)  
                num_preds = len(dt_bboxes)  
                precision = total_tp / num_preds if num_preds > 0 else 0.0               
                final_reward = modify_list_([precision])[0]
                rewards.append(final_reward)

            except Exception as e:
                print(f"{e}")
                rewards.append(0.0)
        except:
            rewards.append(0.0)

    return rewards


###  reward registry three parts
reward_funcs_registry = {
    
    "percept_format":json_format_reward,#-1——0
    "recall_reward": recall_reward,#0——1
    "precision_reward": classification_accuracy_reward,#0——1
    "spatial_precision_reward": spatial_precision_reward,#0——1
    "format_reward":format_reward,#-1——0
    "think_accuracy": think_accuracy, #0——1
    "answer_accuracy":answer_accuracy#0——1
}
# 


def main(script_args, training_args, model_args,args):
    
   
    # Get reward functions
    script_args.reward_funcs = ['percept_format','recall_reward','precision_reward','format_reward','think_accuracy','spatial_precision_reward',"answer_accuracy"]#,
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

   
    dataset=load_from_disk(script_args.dataset_name)
    
    dataset = dataset.shuffle(seed=42)  
   
    # Format into conversation
    def make_conversation(example):
      
        return {
            "prompt": [
              
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        content=[]
        example["problem"]=example["problem"].replace("<image>","")
      
        for img in example["image"]:
                content.append({"type": "image"})
                
        content.append({"type": "text", "text": example["problem"] })
        message={
            "prompt": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }
      
        return message

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        train_dataset = dataset["train"].map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        train_dataset = dataset["train"].map(make_conversation)
        train_dataset = dataset["train"].remove_columns("messages")
    
    
    trainer_cls = Qwen25VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
  

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        adapative=script_args.adaptive,

        torch_dtype=model_args.torch_dtype
    )

    # Train and push the model to the Hub
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
   
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.dataset_name=args.data_path
    ###
    model_args.model_name_or_path=args.base_model
    model_args.use_peft=args.use_peft
    model_args.lora_r=args.lora_r
    model_args.lora_alpha=args.lora_alpha
    model_args.lora_dropout=args.lora_dropout
    model_args.lora_target_modules= args.lora_target_modules
   
    ###
    training_args.num_train_epochs=args.epochs
    training_args.num_generations=args.num_generations
    training_args.beta=args.KL_penaly
    training_args.temperature=args.temperature
    training_args.top_p=args.top_p
    training_args.top_k=args.top_k
    training_args.max_completion_length=args.max_completion_length
    training_args.max_prompt_length=args.max_prompt_length
    training_args.output_dir=args.model_save_path
    training_args.per_device_train_batch_size=args.batch_size
    training_args.learning_rate=args.learning_rate
    training_args.gradient_accumulation_steps=args.gradient_accumulation_steps
    
    training_args.ddp_find_unused_parameters=False
    
    main(script_args, training_args, model_args, args)

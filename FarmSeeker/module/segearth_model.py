import os
import cv2
import torch
from enum import Enum
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
from segearth_process import predict_single_image
from segearth_r1.model.builder import load_pretrained_model
from segearth_r1.mm_utils import get_model_name_from_path
from torch.utils.data import DataLoader
from typing import Optional,List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import json

from typing import Optional

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    base_data_path: str = "/opt/data/private/data_set/fibo_data"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    model_path: Optional[str] = field(default="/opt/data/private/SegEarth-R1-main/checkpoint/SegEarth-R1_ReasonSeg")
    mask_config: Optional[str] = field(default="model/segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r1'
    version: str = 'llava_phi'
    segmentation: bool = True
    eval_batch_size: int = 5
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    data_split: Optional[str] = field(default="val")
    use_seg_query: bool = False
    dataset_type: Optional[str] = field(default="RefSegRS")
    vis_path: Optional[str] = field(default="/opt/data/private/SegEarth-R1-main/test_Cambodia/predict_label")
class BaseSegModel():
    def __init__(self,data_args: Optional[DataArguments] = None):
        self.data_args = data_args if data_args is not None else DataArguments()
        self.model_path=self.data_args.model_path
        
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
        self.data_args.model_path, None,  self.model_name,
        model_args=self.data_args,
        mask_config=self.data_args.mask_config,
        use_seg_query=self.data_args.use_seg_query,
        device='cuda'
    )
    def evaluate(self,image_np):
        pred_mask= predict_single_image(image_np, self.model, self.tokenizer, self.data_args, device='cuda')
        return pred_mask

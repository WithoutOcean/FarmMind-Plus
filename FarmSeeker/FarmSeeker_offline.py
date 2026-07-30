import argparse
import os
import sys
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F
import glob
import time
import csv
from dataclasses import fields
from PIL import Image
from module import save_data
from flask import Flask, request, jsonify
import requests
import base64
from io import BytesIO
from utils.Evaluation_Metrics import calculate_all_metrics
from module import segearth_model




def parse_args():
    parser = argparse.ArgumentParser(description="FarmSeeker Segment")
    parser.add_argument("--base_seg", default="",help="segearth-R1 model weight")
    parser.add_argument("--base_dir", default="",help="all dataset dir", type=str)
    parser.add_argument("--url", default="http://127.0.0.1:7899/chat")
    parser.add_argument("--metrics_path", default="",help="save metrics")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    return parser.parse_args()


base_seg_model_instance = None
def init_model(args):
    global base_seg_model_instance
    data_arg_names = {f.name for f in fields(segearth_model.DataArguments)}
    data_kwargs = {k: v for k, v in vars(args).items() if k in data_arg_names}
    data_kwargs["model_path"] = args.base_seg
    data_args = segearth_model.DataArguments(**data_kwargs)
    base_seg_model_instance = segearth_model.BaseSegModel(data_args)
def main():
    global base_seg_model_instance, args
    gts = []
    preds = []
    corrects = []
    
    with torch.no_grad():
        image_paths = glob.glob(
            os.path.join(args.base_dir, "images", "*.tif")
        )
        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            mask_path = os.path.join(
                args.base_dir,
                "labels",
                image_name.replace(".tif", ".png"),
            )
            pred_mask = base_seg_model_instance.evaluate(img_path)
            if torch.is_tensor(pred_mask):
                pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = np.squeeze(pred_mask)
            pred_mask = pred_mask > 0

            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            if pred_mask.shape != image_np.shape[:2]:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (image_np.shape[1], image_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            mask = Image.open(mask_path)
            masks = np.array(mask)
            if masks.ndim == 3:
                masks = np.max(masks, axis=2)
            gts.append(masks)
            masks = masks[np.newaxis, :, :]
            truth_mask = masks.copy()
            truth_mask[truth_mask != 0] = 1
            overlay_img = image_np.copy()
            overlay_alpha = 0.3
            overlay_color = np.array([0, 255, 0], dtype=np.float32)
            overlay_img[pred_mask] = np.clip(
                image_np[pred_mask].astype(np.float32) + overlay_alpha * overlay_color,
                0,
                255,
            ).astype(np.uint8)
            overlay_img = Image.fromarray(overlay_img)

            truth_mask_list = truth_mask[0].tolist()
            predict_label = pred_mask.astype(int).tolist()
            preds.append(pred_mask.astype(int))
            buffered = BytesIO()
            overlay_img.save(buffered, format="PNG")
            overlay_img_base = base64.b64encode(buffered.getvalue()).decode("utf-8")

            red_layer = np.zeros_like(image_np, dtype=np.uint8)
            red_layer[pred_mask] = [0, 255, 0]
            mask_layer = cv2.cvtColor(red_layer, cv2.COLOR_RGB2BGR)
            mask_layer = Image.fromarray(mask_layer)
            buffered = BytesIO()
            mask_layer.save(buffered, format="PNG")
            mask_layer_base = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response = requests.post(
                args.url,
                json={
                    "overlay_img": overlay_img_base,
                    "pred_mask": mask_layer_base,
                    "image_path": img_path,
                
                    "labels": predict_label,
                    "truth_mask": truth_mask_list,
                },
            ).json()

            last_label = np.array(response["response"]["reasoning_correction"])
            last_label[last_label > 1] = 1
            last_label[last_label < 0] = 0
            corrects.append(last_label.astype(int))
            
    calculate_all_metrics(gts, preds, corrects, args.metrics_path)


if __name__ == "__main__":
    args = parse_args()
    init_model(args)
    main()

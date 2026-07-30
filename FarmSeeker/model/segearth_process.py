import cv2
import torch
import os
from PIL import Image
import numpy as np
import re
from segearth_r1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX , REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from segearth_r1 import conversation as conversation_lib
# 假设你已定义这些函数（如果没有，请补充）
def preprocess_image(image, image_size=1024):
    """将图像缩放到指定大小，并转为 tensor"""
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 填充到 image_size x image_size
    padded = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded.transpose(2, 0, 1)  # HWC -> CHW

def preprocess_mask(mask, original_size):
    """将 mask 缩放到 1024x1024"""
    h, w = mask.shape
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((1024, 1024), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return torch.tensor(padded).long()

def preprocess_referring_instruction(instruction, tokenizer, REFER_token='[SEG]'):
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    tokenized = tokenized + [tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

    token_refer_id = torch.tensor(tokenized)

    return token_refer_id

def tokenizer_special_tokens(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                seg_token_index=SEG_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, answer_token_index = ANSWER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<refer>':refer_token_index, '<answer>':answer_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<refer>|<answer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

def preprocess_llama2(sources, tokenizer):
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_phi']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    input_ids = torch.stack(
        [tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_special_tokens(rou, tokenizer))
            instruction_len = len(tokenizer_special_tokens(parts[0], tokenizer)) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )
   
# 全局常量（与 RefSegRSDataset 一致）
pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
REFER_TOKEN_INDEX = -204  # 请替换为你实际使用的 refer token id
def predict_single_image(
    image_np: str,
    model,
    tokenizer,
    data_args,
    device='cuda',
    image_size=1024
):
    """
    输入单张图像路径，返回预测 mask 和原始图像名
    """
    # 1. 加载并预处理图像
    if isinstance(image_np, str):
        image = cv2.imread(image_np)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image=image_np
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_np}")
    
    processed_image = preprocess_image(image, image_size)
    processed_image = (torch.tensor(processed_image) - pixel_mean) / pixel_std
    processed_image = processed_image.unsqueeze(0).to(device)  # [1, 3, 1024, 1024]

    # 2. 构造文本输入
    ref = "Please segment the farmland in the image"
    prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
    instruction = ' {}'.format(ref)
    sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                {'from': 'gpt', 'value': '\nSure. It is <seg>. '}]]
    
    text_dict = preprocess_llama2(sources, tokenizer)
    input_ids = text_dict['input_ids'][0].unsqueeze(0).to(device)  # [1, seq_len]
    labels = text_dict['labels'][0].unsqueeze(0).to(device)

    token_refer_id = preprocess_referring_instruction(instruction, tokenizer).to(device)
    refer_embedding_indices = torch.zeros_like(input_ids)
    refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
    
    # 3. 构造输入字典（模仿 dataset __getitem__ 输出）
    inputs = {
        'input_ids': input_ids,
        'attention_mask': (input_ids != tokenizer.pad_token_id).long(),
        'images': processed_image,
        'masks': torch.zeros(1, 512, 512).long().to(device),  # dummy mask
        'token_refer_id': [token_refer_id],
        'refer_embedding_indices': refer_embedding_indices,
        'labels': labels,
      
    }
    model.to(device=device,dtype=torch.float).eval()
    # 4. 模型推理
    with torch.no_grad():
        outputs = model.eval_seg(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'].float(),
            masks=inputs['masks'],
            token_refer_id=inputs['token_refer_id'],
            refer_embedding_indices=inputs['refer_embedding_indices'],
            labels=inputs['labels'],
            token_answer_id=None,
            answer_embedding_indices=None
        )

    # 5. 返回预测 mask（第一个样本）
    pred_mask = outputs[0]['pred_masks']  # shape: [num_classes, H, W] 或 [H, W]
    return pred_mask
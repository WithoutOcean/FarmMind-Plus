import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class Qwen3:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to("cuda:3")
        self.model.eval()
    # def chat(self,question):
    #     prompt = question
    #     messages = [
    #     {"role": "user", "content": prompt}
    #     ]
    #     text = self.tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    #     )
    #     model_inputs =  self.tokenizer([text], return_tensors="pt").to(self.model.device)

    #     # conduct text completion
    #     with torch.no_grad():
    #         generated_ids = self.model.generate(
    #         **model_inputs,
    #         max_new_tokens=32768
    #         )
    #         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()   
    #         response=self.tokenizer.decode(output_ids, skip_special_tokens=True)      
    #     return response
    def chat(self,questions):
        texts = []
        
        max_new_tokens=1024
        for question in questions:
            messages = [{"role": "user", "content": question}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            texts.append(text)
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,  # 启用填充
            truncation=True,  # 启用截断
            
            return_attention_mask=True
        ).to(self.model.device)
        
        

        # conduct text completion
        with torch.no_grad():
           
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                
            )
            # 提取生成的token（跳过输入部分）
           
            responses = self.tokenizer.batch_decode(generated_ids[:, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return responses
from flask import Flask, request, jsonify



import time
import torch





model=Qwen3("/g0001sr/why/Qwen3-32B")

app=Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    start=time.time()
    body=request.get_json()
    question=body.get("question")
    response=model.chat(questions=question)
    cost=time.time()-start
    return jsonify(
        {
            "question":question,
            "response":response, 
            "cost":cost
        }
    )


if __name__=="__main__":
    #app.run (host="0.0.0.0",port=30173,processes=5)
    from werkzeug.serving import ThreadedWSGIServer
    server = ThreadedWSGIServer("0.0.0.0", 7892, app)
    server.max_connections = 20 

    server.serve_forever()

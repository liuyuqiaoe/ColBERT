import json
import time
from PIL import Image
import io
import base64
from io import BytesIO

import copy
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch


from typing import Optional, List
from random import sample
import sys, types

# from openai import OpenAI

# from transformers import AutoProcessor, AutoModel, LlavaForConditionalGeneration

import shutil

from transformers import AutoProcessor, AutoModel, AutoTokenizer, LlavaOnevisionForConditionalGeneration

LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
OPENAI_API_KEY = "YOUR_API_KEY"


class GPTGenerator:
    def __init__(self, api_key=OPENAI_API_KEY, model_name="gpt-4o-mini"):
        
        self.model_name = model_name
        print(f"Initializing GPT API with model: {model_name}")
        
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please provide OPENAI_API_KEY environment variable or pass api_key parameter")
            self.client = OpenAI(api_key=api_key)
        
        print("GPT API client initialized successfully!")
    
    def _image_to_base64(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def generate_answer(self, prompt_question_part, prompt_instruction_part, images, max_retries=3):
       
        text_content = f"{prompt_instruction_part}{prompt_question_part}"
        
        content = [{"type": "text", "text": text_content}]
        
        for img in images:
            base64_image = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=256,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content.strip()
                return answer
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded after {max_retries} retries. Error: {error_msg}")
                        raise e
                
                else:
                    raise e


class LLaVAOneVisionQwen2Generator:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        
        print(f"Loading LLaVA-OneVision-Qwen2 model: {model_name}")
    
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 
        )

        print("LLaVA-OneVision-Qwen2 model loaded successfully!")
        
    def generate_answer(self, prompt_question_part, prompt_instruction_part, images):
        text_content = f"{prompt_instruction_part}{prompt_question_part}"
        content = [{"type": "text", "text": text_content}]
        for img in images:
            content.append({
                "type": "image"
            })
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors='pt').to("cuda", torch.float16)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        del inputs
        
        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
        print(answer)
        del outputs
        torch.cuda.empty_cache()
        return answer

    def _image_to_base64(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

def load_ans_file():
    ans_file_path = os.path.join(os.getcwd(), "colbert/tests/retrival.jsonl")
    data = []
    try:
        with open(ans_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: ans file does not exist")
        return
    output_file = open(os.path.join(os.getcwd(),"colbert/tests/answer_file_test.jsonl"), "w")
    idx = 0
    generator = LLaVAOneVisionQwen2Generator()
    for item in data:
        # if idx >=6:
        #     break
        image_path = item["image"]
        retrieved_image_paths = item["retrieved_images"]
        image_files = [image_path] + retrieved_image_paths
        images = [Image.open(image_file).convert("RGB") for image_file in image_files]
    
        ans = generator.generate_answer(item["prompt_question_part"], item["prompt_instruction_part"], images)
       
        output_file.write(json.dumps({
            "qs_id": item['id'],
            "prompt": item['prompt'],
            "output": ans,
            "gt_answer": item['answer'],
            "shortuuid": item["shortuuid"],
            "model_id": 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
            "gt_choice": item['gt_choice'],
            "scenario": item['scenario'],
            "aspect": item['aspect'],
        }) + "\n")
        output_file.flush()
        idx += 1

def load_ans_file_base():
    ans_file_path = os.path.join(os.getcwd(), "colbert/tests/base.jsonl")
    data = []
    try:
        with open(ans_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: ans file does not exist")
        return
    output_file = open(os.path.join(os.getcwd(),"colbert/tests/answer_file_base.jsonl"), "w")
    generator = LLaVAOneVisionQwen2Generator()
    for item in data:
        # if idx >=6:
        #     break
        image_path = item["image"]
        image_files = [image_path]
        images = [Image.open(image_file).convert("RGB") for image_file in image_files]
    
        ans = generator.generate_answer(item["prompt_question_part"], item["prompt_instruction_part"], images)
       
        output_file.write(json.dumps({
            "qs_id": item['id'],
            "prompt": item['prompt'],
            "output": ans,
            "gt_answer": item['answer'],
            "shortuuid": item["shortuuid"],
            "model_id": 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
            "gt_choice": item['gt_choice'],
            "scenario": item['scenario'],
            "aspect": item['aspect'],
        }) + "\n")
        output_file.flush()

if __name__ == "__main__":
    # before running this script, you need check the following:
    # the environment should be different from that of our main project, in which the dependencies are specified in generator_env.yml
    # the retrieval data is stored in the colbert/tests/retrieval.jsonl file colbert/tests/base.jsonl file
    # the input image and gt images in MragBenchmark dataset are restored in colbert/tests/images or colbert/tests/images_base
    # if you don't have the retrieval data and images, you can run the mrag_test_retrieval.py to create them
    # the rate limit error due to the openai api has not been fixed yet, just ignore it
    # if the gpu is available, you can use the LLaVAOneVisionQwen2Generator to generate the answers and avoid the rate limit error
    # after running this script, you will get the answer_file.jsonl that stores the responses of generator and metadata for each question

    load_ans_file()
    # load_ans_file_base()
    
    

# class mPLUGGenerator:
#     def __init__(self, model_name="mPLUG/mPLUG-Owl3-7B-240728"):
        
#         print(f"Loading mPLUG-Owl3 model: {model_name}")
#         sys.modules["icecream"] = types.ModuleType("icecream")
#         sys.modules["icecream"].ic = lambda *x, **k: None
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             device_map="auto",
#             attn_implementation="sdpa",
#             torch_dtype=torch.float16 
#         ).eval()
#         self.model.vision_model.config._attn_implementation = "sdpa"
#         self.processor = self.model.init_processor(self.tokenizer)
#         messages = [
#             {"role": "user", "content": """<|image|>
#         Describe this image."""},
#             {"role": "assistant", "content": ""}
#         ]
#         image = Image.new('RGB', (500, 500), color='red')
#         inputs = self.processor(messages, images=[image], videos=None)

#         inputs.to('cuda')
#         inputs.update({
#             'tokenizer': self.tokenizer,
#             'max_new_tokens':100,
#             'decode_text':True,
#         })


#         g = self.model.generate(**inputs)
#         print(g)
#         print("Loading mPLUG-Owl3 model loaded successfully!")
        
#     def generate_answer(self, prompt_question_part, prompt_instruction_part, images):
#         text_content = f"{prompt_instruction_part}{prompt_question_part}"
#         content = [{"type": "text", "text": text_content}]
#         for img in images:
#             content.append({
#                 "type": "image"
#             })
#         conversation = [{"role": "user", "content": content}]
#         prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
#         inputs = self.processor(images=images, text=prompt, return_tensors='pt').to(torch.float16)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=32,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 pad_token_id=self.processor.tokenizer.eos_token_id
#             )
#         del inputs
        
#         response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         if "Answer:" in response:
#             answer = response.split("Answer:")[-1].strip()
#         else:
#             answer = response
#         print(answer)
#         del outputs
#         torch.cuda.empty_cache()
#         return answer

  



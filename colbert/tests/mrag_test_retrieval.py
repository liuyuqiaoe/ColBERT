import json
import time
from datasets import load_dataset
from MRAG.eval.utils.dataloader import bench_data_loader 
from tqdm import tqdm
from PIL import Image
import io
import base64
from io import BytesIO

from colbert.infra.config import ColBERTConfig
from colbert.searcher import HFSearcher

import copy
import torch

import os
from colbert.modeling.checkpoint import HFCheckpoint
from pathlib import Path
from typing import Optional, List
from random import sample
from colbert.data.collection import ImageCollection
from colbert.indexer import ImageIndexer
from colbert.parameters import DEVICE

from openai import OpenAI

from transformers import AutoProcessor, LlavaForConditionalGeneration

import shortuuid
import shutil


def get_image_paths(root_dir: Optional[str] = None):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_paths = []
    if root_dir is None or root_dir == '':
        root_dir = os.path.join(os.getcwd(), "data/image_data")

    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if Path(file_path).suffix.lower() in image_extensions:
                if os.path.isfile(file_path):
                    image_paths.append(file_path)
    return image_paths

def sample_image_paths(image_paths: List[str], num_samples: int = 7000):
    return sample(image_paths, num_samples)

def get_image_collection():
    root_dir = os.path.join(os.getcwd(), "data/image_data")
    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    image_paths = get_image_paths(root_dir)
    image_paths = sample_image_paths(image_paths, 4000)

    collection = ImageCollection(image_paths=image_paths)

    collection_path = os.path.join(os.getcwd(), "colbert/tests/image_collection.txt")

    if os.path.exists(collection_path):
        print(f"Removing existing collection file at {collection_path}")
        os.remove(collection_path)

    collection.save(collection_path)
    print(f"Collection saved to {collection_path}")
    
    return collection

def load_image_collection(collection_path: str = None):
    if collection_path is None or collection_path == "" or not os.path.exists(collection_path):
        collection_path = os.path.join(os.getcwd(), "colbert/tests/image_collection.txt")
    collection = ImageCollection(path=collection_path)
    return collection

def test_image_indexer_indexing(collection):
    print("Testing ImageIndexer full indexing...")
    
    print(f"Created collection with {len(collection)} images")
    
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")

    if not os.path.exists(index_path):
        os.makedirs(index_path)
       
    config = ColBERTConfig(
        index_path=index_path,
        index_bsize=3,
        # dim=512,  # CLIP model dimension
        nranks=1,
        rank=0,
        resume=False,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
    )
    
    # Create indexer
    indexer = ImageIndexer(config=config, verbose=3)
    print("Created ImageIndexer")

    # Test indexing
    print("Starting indexing...")
    index_path = indexer.index(
        name="test_indexing",
        collection=collection,
        overwrite=True
    )
    
    print(f"Index built successfully at: {index_path}")
    
    # Check index files
    index_files = os.listdir(index_path)
    print(f"Created {len(index_files)} index files:")
    
    # Test get_index method
    retrieved_path = indexer.get_index()
    print(f"get_index() returned: {retrieved_path}")
    
    print("Full indexing test passed!")
    

def bench_data_loader(mode,image_placeholder="<image>"):
    mode_lst = ["base", "using_gt_images", "using_retrieved_examples", "using_clip_retriever"]
    assert mode in mode_lst
    # Data
    mrag_bench = load_dataset("uclanlp/MRAG-Bench",split="test")
    
    for item in tqdm(mrag_bench):
        
        qs_id = item['id'] 
        qs = item['question']
        ans = item['answer']
        gt_choice = item['answer_choice']
        scenario = item['scenario']
        choices_A = item['A']
        choices_B = item['B']
        choices_C = item['C']
        choices_D = item['D']
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images]
        
        image = item['image'].convert("RGB") 

        if scenario == 'Incomplete':
            gt_images = [gt_images[0]]        

        if mode == "base":
            prompt = f"Answer with the option's letter from the given choices directly.\n"
            image_files = [image]
        elif mode == "using_gt_images":
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image] + gt_images
            if scenario == "Incomplete":
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
        elif mode == "using_retrieved_examples":
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            retrieved_images = item["retrieved_images"]
            retrieved_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib["bytes"])) for ib in retrieved_images]
            if scenario == "Incomplete":
                retrieved_images = [retrieved_images[0]]
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image] + retrieved_images
        else:
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image]
        
        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        prompt_question_part = qs
        prompt_instruction_part = prompt
        qs = prompt + qs
        
        yield {
            "id": qs_id, 
            "question": qs, 
            "image_files": image_files, 
            "prompt": qs,
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "prompt_question_part": prompt_question_part,
            "prompt_instruction_part": prompt_instruction_part,
            "aspect": item['aspect'],
            "gt_images": gt_images,
            "plain_question": item["question"],
            "choices": [["A", choices_A], ["B", choices_B], ["C", choices_C], ["D", choices_D]]
        }

def get_embeddings_approach1(searcher, qs, choices, const=0.25):
    tmp_choices = copy.deepcopy(choices)
    qs_embedding = searcher.encode(qs)
    choice_text = [item[1] for item in tmp_choices]
    choices_embedding = searcher.encode(choice_text)
    choice_embeddings = list(torch.split(choices_embedding, 1, dim=0))
    
    assert len(tmp_choices) == len(choice_embeddings)
    assert choice_embeddings[0].size(0) == 1 and choice_embeddings[0].size(1) == 1

    for i in range(len(tmp_choices)):
        tmp_choices[i].append(const * qs_embedding + (1 - const) * choice_embeddings[i])
    
    return tmp_choices

def get_embeddings_approach2(searcher, qs, choices):
    tmp_choices = copy.deepcopy(choices)
    qs_choice_text = [f"Question:{qs}\nThis is the choice {item[0]}:{item[1]}" for item in tmp_choices]
    qs_choices_embedding = searcher.encode(qs_choice_text)
    qs_choice_embeddings = list(torch.split(qs_choices_embedding, 1, dim=0))

    assert len(tmp_choices) == len(qs_choice_embeddings)
    assert qs_choice_embeddings[0].size(0) == 1 and qs_choice_embeddings[0].size(1) == 1

    for i in range(len(tmp_choices)):
        tmp_choices[i].append(qs_choice_embeddings[i])

    return tmp_choices

def get_diverse_retrieved_images(tmp_choices):
    choices_num = len(tmp_choices)
    current_idx = [0] * choices_num
    selected_pids = [[i, item[3]["pids"][0], item[3]["scores"][0]] for i, item in enumerate(tmp_choices)]
    selected_pids = sorted(selected_pids, key=lambda item: item[2], reverse=True)

    if len(set([item[1] for item in selected_pids])) == choices_num:
        return selected_pids
    
    selected_pids_set = set()
    i = 0
    while i < choices_num:
       
        if selected_pids[i][1] not in selected_pids_set:
            selected_pids_set.add(selected_pids[i][1])
            i += 1
        else:
            current_idx[selected_pids[i][0]] += 1
            selected_pids[i][1] = tmp_choices[selected_pids[i][0]][-1]["pids"][current_idx[selected_pids[i][0]]]
            selected_pids[i][2] = tmp_choices[selected_pids[i][0]][-1]["scores"][current_idx[selected_pids[i][0]]]

    selected_pids = sorted(selected_pids, key=lambda item: item[0])
    return selected_pids

def get_retrieval_images(searcher, qs, choices, approach_num=1):
    assert approach_num in [1, 2]

    if approach_num == 1:
        tmp_choices = get_embeddings_approach1(searcher, qs, choices)
    else:
        tmp_choices = get_embeddings_approach2(searcher, qs, choices)

    for item in tmp_choices:
        pids, ranks, scores = searcher.search_emb_input(Q=item[2], k=4) # k must be equal to orlarger than the number of choices
        item.append({"pids": pids, "ranks": ranks, "scores": scores})
        # item.append(pids[0])
        assert len(item) == 4

    selected_pids = get_diverse_retrieved_images(tmp_choices)
    
    return selected_pids

def dump_images(image, gt_images, item_id, output_dir):
    sub_dir = os.path.join(output_dir, f"question_{item_id}")
    if os.path.exists(sub_dir):
        print(f"remove existing path {sub_dir}")
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir)
    input_image_path = os.path.join(sub_dir, "input_image.jpg")
    image.save(input_image_path)
    gt_dir = os.path.join(sub_dir, f"gt_images")
    os.makedirs(gt_dir, exist_ok=True)
    gt_image_paths = []
    for i, gt_image in enumerate(gt_images):
        file_name = f"gt_image_{i}.jpg"
        gt_image_path = os.path.join(gt_dir, file_name)
        gt_image_paths.append(gt_image_path)
        gt_image.save(gt_image_path)
    return input_image_path, gt_image_paths


def test_retrieval_mragbenchmark(collection):
    mode = "using_clip_retriever"

    print("Testing retrieval on MRAG benchmark")
   
    index_path = os.path.join(os.getcwd(),"colbert/tests/indexes")
    index_plan_path = os.path.join(index_path, "plan.json")

    try:
        with open(index_plan_path, "r") as f:
            checkpoint_plan = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Plan file does not exist")
        return
    
    collection_config = checkpoint_plan["config"]
    print("Loaded config: \n", collection_config)
    config = ColBERTConfig()
    config.configure(**collection_config)
    config.configure(ncells=4)
    searcher = HFSearcher(config=config, verbose=1, collection=collection)
    print("HFSearcher created")

    ans_file = open(os.path.join(os.getcwd(),"colbert/tests/retrival.jsonl"), "w")
    searcher = HFSearcher(config=config, verbose=1)
    # gpt_generator = GPTGenerator()
    for item in bench_data_loader(mode):

        qs = item["plain_question"]
        choices = item["choices"]
        selected_pids = get_retrieval_images(
            searcher=searcher, 
            qs=qs, 
            choices=choices, 
            approach_num=2
            )
        
        selected_image_paths = [collection[item[1]] for item in selected_pids]
        
        output_dir = os.path.join(os.getcwd(), "colbert/tests/images")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        input_image_path, gt_image_paths = dump_images(item["image_files"][0], item["gt_images"], item['id'], output_dir)
        ans_file.write(json.dumps({
            "id": item['id'],
            "question": item["question"],
            "prompt": item['prompt'],
            "answer": item["answer"],
            "gt_choice": item["gt_choice"],
            "prompt_question_part": item["prompt_question_part"],
            "prompt_instruction_part": item["prompt_instruction_part"],
            "image": input_image_path,
            "gt_images": gt_image_paths,
            "retrieved_images": selected_image_paths,
            "gt_answer": item['answer'],
            "shortuuid": shortuuid.uuid(),
            "approach": '2',
            "scenario": item['scenario'],
            "aspect": item['aspect'],
        }) + "\n")
        ans_file.flush()

def test_retrieval_mragbenchmark_base(collection):
    mode = "base"

    print("Testing retrieval on MRAG benchmark (base)")

    ans_file = open(os.path.join(os.getcwd(),"colbert/tests/base.jsonl"), "w")
   
    # gpt_generator = GPTGenerator()
    for item in bench_data_loader(mode):
       
        qs = item["plain_question"]
        choices = item["choices"]
        
        output_dir = os.path.join(os.getcwd(), "colbert/tests/images_base")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        input_image_path, gt_image_paths = dump_images(item["image_files"][0], item["gt_images"], item['id'], output_dir)
        ans_file.write(json.dumps({
            "id": item['id'],
            "question": item["question"],
            "prompt": item['prompt'],
            "answer": item["answer"],
            "gt_choice": item["gt_choice"],
            "prompt_question_part": item["prompt_question_part"],
            "prompt_instruction_part": item["prompt_instruction_part"],
            "image": input_image_path,
            "gt_images": gt_image_paths,
            "gt_answer": item['answer'],
            "shortuuid": shortuuid.uuid(),
            "scenario": item['scenario'],
            "aspect": item['aspect'],
        }) + "\n")
        ans_file.flush()
        

if __name__ == "__main__":
    # before running this script, you need check the following:
    # the index is created in the colbert/tests/indexes directory
    # the collection is created in the colbert/tests/image_collection.txt file
    # the image_collection.txt stores an image path per line
    # if you don't have the index and collection, you can run the image_indexer_test.py to create them
    # the rate limit error due to the openai api has not been fixed yet, just ignore it
    # if the gpu is available, you can use the llava model to generate the answers and avoid the rate limit error
    # the llava model way is not tested yet

    # test_image_collection()
    collection = load_image_collection()
    # test_image_indexer(collection)
    # test_retrieval_mragbenchmark(collection)
    test_retrieval_mragbenchmark_base(collection)
   
    









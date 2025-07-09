import sys
import os
import torch
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List
from random import sample
from colbert.data.collection import ImageCollection
from colbert.indexing.image_collection_indexer import ImageCollectionIndexer
from colbert.infra.config import ColBERTConfig

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
    # get image paths from the root directory
    root_dir = os.path.join(os.getcwd(), "data/image_data")
    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    image_paths = get_image_paths(root_dir)
    image_paths = sample_image_paths(image_paths, 10)

    # Create collection
    collection = ImageCollection(image_paths=image_paths)
    return collection

def test_image_collection_indexer_basic(collection):
    print("Testing ImageCollectionIndexer (basic)...")
    
    print(f"Collection created with {len(collection)} images")
    
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    config = ColBERTConfig(
        index_path=index_path,
        index_bsize=3,
        dim=512,  # CLIP embedding dimension
        nranks=1,
        rank=0,
        resume=False,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=128,
        hf_default_batch_size=4
    )
    print("Config created successfully")
    
    # Create indexer
    indexer = ImageCollectionIndexer(config=config, collection=collection, verbose=2)

    print("ImageCollectionIndexer created successfully")
    print(f"  Collection length: {len(indexer.collection)}")
    print(f"  Use GPU: {indexer.use_gpu}")
    print(f"  Rank: {indexer.rank}")
    print(f"  Nranks: {indexer.nranks}")
    
    print("\nBasic ImageCollectionIndexer test completed!")

def test_image_collection_indexer_setup(collection):
    print("Testing ImageCollectionIndexer setup...")

    print(f"Collection created with {len(collection)} images")
    
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    config = ColBERTConfig(
        index_path=index_path,
        index_bsize=3,
        # dim=512,  # CLIP embedding dimension
        nranks=1,
        rank=0,
        resume=False,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=128,
        hf_default_batch_size=4
    )
    print("Config created successfully")
    
    # Create indexer
    indexer = ImageCollectionIndexer(config=config, collection=collection, verbose=2)
    print("ImageCollectionIndexer created successfully")
    
    # Test setup
    try:
        print("\nTesting setup...")

        indexer.setup()
        print("Setup completed successfully!")
        print(f"  Num chunks: {indexer.num_chunks}")
        print(f"  Num partitions: {indexer.num_partitions}")
        print(f"  Num embeddings est: {indexer.num_embeddings_est}")
        print(f"  Avg doclen est: {indexer.avg_doclen_est}")
        
        # Check if plan file was created
        plan_path = os.path.join(config.index_path_, 'plan.json')
        if os.path.exists(plan_path):
            print(f"  Plan file created: {plan_path}")
        else:
            print("  Plan file not found")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        shutil.rmtree(index_path)
        print(f"Cleaned up {index_path}")
                

def test_image_indexer_run(collection):
    # TODO: collection size
    print("Testing ImageCollectionIndexer.run()...")
    
    print(f"Collection created with {len(collection)} images")
    
    # Create index path
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    config = ColBERTConfig(
        index_path=index_path,
        index_bsize=3,
        dim=512,  # CLIP embedding dimension
        nranks=1,
        rank=0,
        resume=False,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=128,
        hf_default_batch_size=4
    )
    print("Config created successfully")
    
    # Create indexer
    indexer = ImageCollectionIndexer(config=config, collection=collection, verbose=2)
    print("ImageCollectionIndexer created successfully")
    
    # Test the run method
    try:
        shared_lists = []  # Empty for single-process (cpu only)
        indexer.run(shared_lists)
        
        print("Indexing run completed successfully!")
        
        # Check created files
        index_dir = config.index_path
        if os.path.exists(index_dir):
            files = os.listdir(index_dir)
            print(f"\n Created {len(files)} index files:")
            for file in sorted(files):
                file_path = os.path.join(index_dir, file)
                size = os.path.getsize(file_path)
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        shutil.rmtree(index_path)
        print(f"Cleaned up {index_path}")

def run_all_tests():
   
    print("Running ImageCollectionIndexer test...\n")
    
    collection = get_image_collection()

    test_image_collection_indexer_basic(collection)
   
    test_image_collection_indexer_sampling(collection)
    
    test_image_collection_indexer_setup(collection)
    
    # test_image_indexer_run(collection)
    
if __name__ == "__main__":
    run_all_tests()
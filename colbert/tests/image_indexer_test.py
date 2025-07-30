import sys
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from pathlib import Path
from typing import Optional, List
from random import sample
from colbert.data.collection import ImageCollection
from colbert.indexer import ImageIndexer
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
    root_dir = os.path.join(os.getcwd(), "data/image_data")
    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    image_paths = get_image_paths(root_dir)
    # image_paths = sample_image_paths(image_paths, 4000)
    print("Image collection length:", len(image_paths))
    collection = ImageCollection(image_paths=image_paths)

    collection_path = os.path.join(os.getcwd(), "colbert/tests/image_collection.txt")

    if os.path.exists(collection_path):
        print(f"Removing existing collection file at {collection_path}")
        os.remove(collection_path)

    collection.save(collection_path)
    print(f"Collection saved to {collection_path}")
    
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
            
if __name__ == "__main__":
    collection = get_image_collection()
    test_image_indexer_indexing(collection)

            
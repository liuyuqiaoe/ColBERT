import torch
import os
from PIL import Image
from colbert.modeling.checkpoint import HFCheckpoint
from colbert.infra.config import ColBERTConfig
from pathlib import Path
from typing import Optional, List
from random import sample
from colbert.data.collection import ImageCollection

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

def test_image_collection():
    # get image paths from the root directory
    root_dir = os.path.join(os.getcwd(), "data/image_data")
    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    image_paths = get_image_paths(root_dir)
    image_paths = sample_image_paths(image_paths, 25)

    collection = ImageCollection(image_paths=image_paths)
    print(f"Collection created with {len(collection)} images")

    return collection

def test_hfcheckpoint_basic_usage():
    print("=" * 60)
    print("HFCheckpoint Test - Basic Usage")
    print("=" * 60)
    
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")
    
    config = ColBERTConfig(
        index_path=index_path,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=64
    )

    print("Config created successfully")
    checkpoint = HFCheckpoint(colbert_config=config, verbose=1)
    
    print(f"Checkpoint created successfully!")
    print(f"   Model: {config.hf_model_name}")
    print(f"   Model type: {config.hf_model_type}")
    print(f"   Device: {checkpoint.device}")

def test_text_encoding():
    print("\n" + "=" * 60)
    print("Text Encoding Test")
    print("=" * 60)
    
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")
    config = ColBERTConfig(
        index_path=index_path,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=64
    )
    checkpoint = HFCheckpoint(colbert_config=config, verbose=1)
    
    # Sample texts
    texts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A delicious pizza with cheese and pepperoni",
        "A modern city skyline at night",
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A delicious pizza with cheese and pepperoni",
        "A modern city skyline at night",
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A delicious pizza with cheese and pepperoni",
        "A modern city skyline at night",
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A delicious pizza with cheese and pepperoni",
        "A modern city skyline at night"
    ]
    
    print("Sample texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    # Test different keep_dims modes
    print("\n1. keep_dims=True (3D tensor):")
    result = checkpoint.docFromText(texts, keep_dims=True, to_cpu=True, bsize=5)
    embeddings = result[0]
    print(f"   Shape: {embeddings.shape}")
    
    print("\n2. keep_dims='flatten' (2D tensor + doclens):")
    result = checkpoint.docFromText(texts, keep_dims="flatten", to_cpu=True, bsize=5)
    embeddings, doclens = result
    print(f"   Shape: {embeddings.shape}")
    print(f"   Doclens: {doclens}")
    
    print("\n3. keep_dims=False (list of tensors):")
    result = checkpoint.docFromText(texts, keep_dims=False, to_cpu=True, bsize=5)
    embeddings_list = result[0]
    print(f"   Number of tensors: {len(embeddings_list)}")
    for i, emb in enumerate(embeddings_list):
        print(f"   Doc {i+1}: shape {emb.shape}")

def test_image_encoding():
    print("\n" + "=" * 60)
    print("Image Encoding Test")
    print("=" * 60)

    collection = test_image_collection()
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")

    config = ColBERTConfig(
        index_path=index_path,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=64
    )
    checkpoint = HFCheckpoint(colbert_config=config, verbose=1)
    
    image_paths = collection.data
    
    # Test different keep_dims modes
    print("\n1. keep_dims=True (3D tensor - matching text behavior):")
    result = checkpoint.imageFromPath(image_paths, keep_dims=True, to_cpu=True, bsize=5)
    embeddings = result[0]
    print(f"   Shape: {embeddings.shape}")
    
    print("\n2. keep_dims='flatten' (2D tensor + doclens):")
    result = checkpoint.imageFromPath(image_paths, keep_dims="flatten", to_cpu=True, bsize=5)
    embeddings, doclens = result
    print(f"   Shape: {embeddings.shape}")
    print(f"   Doclens: {doclens}")
   
    print("\n3. keep_dims=False (list of tensors):")
    result = checkpoint.imageFromPath(image_paths, keep_dims=False, to_cpu=True, bsize=5)
    embeddings_list = result[0]
    print(f"   Number of tensors: {len(embeddings_list)}")
    for i, emb in enumerate(embeddings_list):
        print(f"   Image {i+1}: shape {emb.shape}")

def test_consistency():
    print("\n" + "=" * 60)
    print("Text-Image Consistency Test")
    print("=" * 60)
    
    collection = test_image_collection()
    index_path = os.path.join(os.getcwd(), "colbert/tests/indexes")

    config = ColBERTConfig(
        index_path=index_path,
        hf_model_name="openai/clip-vit-base-patch16",
        hf_model_type="clip",
        hf_text_max_length=64
    )
    checkpoint = HFCheckpoint(colbert_config=config, verbose=1)
    
    # Same number of items
    documents = [
        "A beautiful sunset over the ocean with golden light",
        "A cat sitting on a windowsill watching birds",
        "A delicious pizza with melted cheese and pepperoni",
        "A modern city skyline illuminated at night"
    ]
    image_paths = collection[:4]
    
    print("Comparing text and image encoding with same number of items:")
    print(f"   Texts: {len(documents)} items")
    print(f"   Images: {len(image_paths)} items")
    
    # Compare keep_dims=True
    print("\nkeep_dims=True comparison:")
    text_result = checkpoint.docFromText(documents, keep_dims=True, to_cpu=True)
    image_result = checkpoint.imageFromPath(image_paths, keep_dims=True, to_cpu=True)
    
    text_emb = text_result[0]
    image_emb = image_result[0]
    
    print(f"   Text shape: {text_emb.shape}")
    print(f"   Image shape: {image_emb.shape}")
    
    # Compare keep_dims="flatten"
    print("\nkeep_dims='flatten' comparison:")
    text_result = checkpoint.docFromText(documents, keep_dims="flatten", to_cpu=True)
    image_result = checkpoint.imageFromPath(image_paths, keep_dims=True, to_cpu=True)
    
    text_emb, text_doclens = text_result
    image_emb, image_doclens = image_result
    
    print(f"   Text shape: {text_emb.shape}, doclens: {text_doclens}")
    print(f"   Image shape: {image_emb.shape}, doclens: {image_doclens}")

def main():
    print("HFCheckpoint Test")

    test_hfcheckpoint_basic_usage()
    test_text_encoding()
    test_image_encoding()
    test_consistency()
    
    print("Test completed successfully!")
    
if __name__ == "__main__":
    main() 
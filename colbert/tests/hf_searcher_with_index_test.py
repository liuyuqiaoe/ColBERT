import os
import sys
import torch
from PIL import Image
from colbert.searcher import HFSearcher
from colbert.infra.config import ColBERTConfig
from pathlib import Path
from typing import Optional, List
from random import sample
from colbert.data.collection import ImageCollection
import json
from pathlib import Path
import os

# before running this script, you need check the following:
# the index is created in the colbert/tests/indexes directory
# the collection is created in the colbert/tests/image_collection.txt file
# the image_collection.txt stores an image path per line
# if you don't have the index and collection, you can run the image_indexer_test.py to create them
def main():
    print("Test HFSearcher with existing index")

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

    collection_path = os.path.join(os.getcwd(),"colbert/tests/image_collection.txt")
    collection = ImageCollection(path=collection_path)
    searcher = HFSearcher(config=config, verbose=1, collection=collection)
    print("HFSearcher created")
    
    # Test queries
    queries = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A delicious pizza with cheese"
    ]
    
    print("Testing searches:")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("1. Encoding query...")

        Q = searcher.encode(query)
        print(f"Query encoded successfully: {Q.shape}")
        
        print("2. Performing search...")
        pids, ranks, scores = searcher.search(query, k=3)
        print(f"Search completed successfully!")

        print(f"Found {len(pids)} results:")
        for j, (pid, rank, score) in enumerate(zip(pids, ranks, scores)):
            print(pid)
            print(f"  {j+1}. PID: {pid}, Score: {score:.4f}, Image Path: {collection[pid]}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 
# ColBERT Image Search Project Summary

## Project Overview

This project extends the original ColBERT framework to support **image search functionality** using CLIP models and integrates the **MRAG Benchmark**.

## Key Features Implemented

### 1. **Image Search Capabilities**
- **ImageCollection Class**: Custom collection handler for image paths
- **ImageIndexer**: Specialized indexer for building image search indexes
- **HFSearcher**: Enhanced searcher with CLIP model support
- **HFCheckpoint**: Checkpoint handling for HuggingFace models
- **Image Model Support**: Configuration updates for CLIP models

### 2. **MRAG Evaluation Framework**
- **Answer Generator**: Supports GPT or LLaVA multi-modal models for answer generation
- **Image Retrieval Strategies**: 
  - Approach 1: Generates query embedding using weighted sum of query embedding and choice embedding
  - Approach 2: Generates query embedding by integrating query text and choice text as input

## Files Added/Modified

### Core Features
- `colbert/data/collection.py` - Added ImageCollection class
- `colbert/indexing/collection_encoder.py` - Added collection encoder that supports both image collection and document collection
- `colbert/indexer.py` - Added ImageIndexer
- `colbert/searcher.py` - Added HFSearcher
- `colbert/modeling/checkpoint.py` - Added HFCheckpoint 
- `colbert/indexing/image_collection_indexer.py` - New image indexing module

### MRAG Framework
- `MRAG/` - Original MRAG repository

### Configuration Updates
- `colbert/infra/config/settings.py` - Updated configuration
- `colbert/infra/config/config.py` - Enhanced config options

### Testing Suite
- `colbert/tests/hf_searcher_with_index_test.py` - Main search test
- `colbert/tests/image_indexer_test.py` - Image indexer tests
- `colbert/tests/image_collection_indexer_test.py` - Collection indexer tests
- `colbert/tests/hfcheckpoint_test.py` - Checkpoint tests
- `colbert/tests/mrag_test.py` - MRAG evaluation tests

### Sample Data & Indexes
- `colbert/tests/image_collection.txt` - Sample image paths
- `colbert/tests/gpt_answers.jsonl` - Sample GPT responses
- `colbert/tests/indexes/` - Generated search indexes
- `data/image_data.zip` - Image dataset (large file)

### Environment Configuration
- `environment_cpu_clean.yml` - Clean CPU environment setup

## Testing Results

### Image Search Tests (CPU)
- ImageCollection loading and management: **Pass**
- ImageIndexer index building: **Pass**
- HFSearcher similarity search: **Pass**
- HFCheckpoint functionality: **Pass**

### MRAG Evaluation Tests (CPU)
- Image retrieval strategies: **Pass**, but awaiting performance evaluation and optimization
- GPT answer generation: **Pass**, but encountered rate limit exceeded error
- LLaVA model for answer generation: **Fail**, supporting GPU only

### GPU Testing
- In progress

### MRAG Benchmark Implementation on GPU
- To be completed




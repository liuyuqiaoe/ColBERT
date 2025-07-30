import torch
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Union, List, Optional, Tuple, Dict, Any
from PIL import Image
import os

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.colbert import ColBERT
from colbert.infra.config import ColBERTConfig

from transformers import CLIPProcessor, CLIPModel

from dataclasses import dataclass
from colbert.infra.config.core_config import DefaultVal
from colbert.parameters import DEVICE
from colbert.infra.config.core_config import CoreConfig
from colbert.infra.run import Run

from transformers import CLIPConfig



def pool_embeddings_hierarchical(
    p_embeddings,
    token_lengths,
    pool_factor,
    protected_tokens: int = 0,
    showprogress: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_embeddings = p_embeddings.to(device)
    pooled_embeddings = []
    pooled_token_lengths = []
    start_idx = 0

    T = tqdm(token_lengths, desc="Pooling tokens") if showprogress else token_lengths
    for token_length in T:
        # Get the embeddings for the current passage
        passage_embeddings = p_embeddings[start_idx : start_idx + token_length]

        # Remove the tokens at protected_tokens indices
        protected_embeddings = passage_embeddings[:protected_tokens]
        passage_embeddings = passage_embeddings[protected_tokens:]

        # Cosine similarity computation (vector are already normalized)
        similarities = torch.mm(passage_embeddings, passage_embeddings.t())

        # Convert similarities to a distance for better ward compatibility
        similarities = 1 - similarities.cpu().numpy()

        # Create hierarchical clusters using ward's method
        Z = linkage(similarities, metric="euclidean", method="ward")
        # Determine the number of clusters we want in the end based on the pool factor
        max_clusters = (
            token_length // pool_factor if token_length // pool_factor > 0 else 1
        )
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        # Pool embeddings within each cluster
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(
                torch.tensor(cluster_labels == cluster_id, device=device)
            )[0]
            if cluster_indices.numel() > 0:
                pooled_embedding = passage_embeddings[cluster_indices].mean(dim=0)
                pooled_embeddings.append(pooled_embedding)

        # Re-add the protected tokens to pooled_embeddings
        pooled_embeddings.extend(protected_embeddings)

        # Store the length of the pooled tokens (number of total tokens - number of tokens from previous passages)
        pooled_token_lengths.append(len(pooled_embeddings) - sum(pooled_token_lengths))
        start_idx += token_length

    pooled_embeddings = torch.stack(pooled_embeddings)
    return pooled_embeddings, pooled_token_lengths


class Checkpoint(ColBERT):
    """
    Easy inference with ColBERT.

    TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None, verbose: int = 3):
        super().__init__(name, colbert_config)
        assert self.training is False

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(self.colbert_config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(
        self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False
    ):
        if bsize:
            batches = self.query_tokenizer.tensorize(
                queries,
                context=context,
                bsize=bsize,
                full_length_search=full_length_search,
            )
            batches = [
                self.query(input_ids, attention_mask, to_cpu=to_cpu)
                for input_ids, attention_mask in batches
            ]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context, full_length_search=full_length_search
        )
        return self.query(input_ids, attention_mask)

    def docFromText(
        self,
        docs,
        bsize=None,
        keep_dims=True,
        to_cpu=False,
        showprogress=False,
        return_tokens=False,
        pool_factor=1,
        protected_tokens=0,
        clustering_mode: str = "hierarchical",
    ):
        assert keep_dims in [True, False, "flatten"]
        assert clustering_mode in ["hierarchical"]

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(
                docs, bsize=bsize
            )

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = "return_mask" if keep_dims == "flatten" else keep_dims
            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                for input_ids, attention_mask in tqdm(
                    text_batches, disable=not showprogress
                )
            ]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == "flatten":
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = (
                    torch.cat(D)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()

                if pool_factor > 1:
                    print(f"Clustering tokens with a pool factor of {pool_factor}")
                    D, doclens = pool_embeddings_hierarchical(
                        D,
                        doclens,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                        showprogress=showprogress,
                    )

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = D @ Q
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output


"""
TODO:

def tokenize_and_encode(checkpoint, passages):
    embeddings, token_ids = checkpoint.docFromText(passages, bsize=128, keep_dims=False, showprogress=True, return_tokens=True)
    tokens = [checkpoint.doc_tokenizer.tok.convert_ids_to_tokens(ids.tolist()) for ids in token_ids]
    tokens = [tokens[:tokens.index('[PAD]') if '[PAD]' in tokens else -1] for tokens in tokens]
    tokens = [[tok for tok in tokens if tok not in checkpoint.skiplist] for tokens in tokens]

    return embeddings, tokens

"""

# Tmp_hfsettings class removed - now integrated into ColBERTConfig as HFSettings



class HFCheckpoint:
    def __init__(self, colbert_config: Optional[ColBERTConfig] = None,
                 verbose: int = 3):
       
        self.verbose = verbose
        self.device = DEVICE

        if colbert_config is not None:
            self.config = colbert_config
        else:
            self.config = ColBERTConfig.from_existing(Run().config)

        # TODO: Validate settings
        # self.config.validate()
        
        self.model_name = self.config.hf_model_name
        self.model_type = self.config.hf_model_type

        print()
        # TODO: Set device
        # self.device = self.settings.device_

        self._load_model()
        
        self.amp_manager = MixedPrecisionManager(self.config.hf_amp and self.device == "cuda")
        
        self.training = False
        
        if self.verbose > 1:
            print(f"HFCheckpoint initialized:")
            print(f"  Model: {self.model_name}")
            print(f"  Type: {self.model_type}")
            print(f"  Device: {self.device}")
            print(f"  Embedding dim: {self.embedding_dimension}")

    def _load_model(self):
        self.has_text_encoder = False
        self.has_image_encoder = False
        try:
            if self.model_type == "clip":
                self._load_clip_model()
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
                
        except Exception as e:
            if self.config.hf_fallback_to_clip:
                print(f"Failed to load model {self.model_name}: {e}")
                print(f"Falling back to {self.config.hf_fallback_model_name}...")
                self.model_name = self.config.hf_fallback_model_name
                self.model_type = "clip"
                self._load_clip_model()
            else:
                raise

    def _load_clip_model(self):
        self.model = CLIPModel.from_pretrained(
            self.model_name, 
            **self.config.hf_model_config_
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)                                               
        self.has_text_encoder = True
        self.has_image_encoder = True
        self.embedding_dimension = self.model.projection_dim
        self.config.configure(hf_embedding_dimension=self.embedding_dimension, dim=self.embedding_dimension)

    def _load_other_model(self):
        raise NotImplementedError

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False):
        if not self.has_text_encoder:
            raise ValueError("This model does not support text encoding")
            
        if isinstance(queries, str):
            queries = [queries]
            
        bsize = bsize or self.config.hf_default_batch_size
        bsize = min(bsize, self.config.hf_max_batch_size)
        
        if bsize and len(queries) > bsize:
            all_embeddings = []
            for i in range(0, len(queries), bsize):
                batch_queries = queries[i:i + bsize]
                batch_embeddings, batch_doclens = self._encode_text_batch(batch_queries, to_cpu) # (batch_size, dim)
                all_embeddings.append(batch_embeddings)
            embeddings = torch.cat(all_embeddings)
            # Add token dimension to make it (batch_size, 1, dim) for ColBERT compatibility
            embeddings = embeddings.unsqueeze(1)
            return embeddings
        else:
            embeddings, doclens = self._encode_text_batch(queries, to_cpu)
            # Add token dimension to make it (batch_size, 1, dim) for ColBERT compatibility
            embeddings = embeddings.unsqueeze(1)
            return embeddings

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        if not self.has_text_encoder:
            raise ValueError("This model does not support text encoding")
            
        if isinstance(docs, str):
            docs = [docs]

        assert keep_dims in [True, False, "flatten"]
            
        bsize = bsize or self.config.hf_default_batch_size
        bsize = min(bsize, self.config.hf_max_batch_size)
        
        if bsize and len(docs) > bsize:
            all_embeddings = []
            all_doclens = []
            
            iterator = range(0, len(docs), bsize)
            if showprogress:
                iterator = tqdm(iterator, desc="Encoding documents", disable=self.verbose < 2)
                
            for i in iterator:
                batch_docs = docs[i:i + bsize]
                batch_embeddings, batch_doclens = self._encode_text_batch(batch_docs, to_cpu)
                all_embeddings.append(batch_embeddings)
                all_doclens.extend(batch_doclens)
            
            if keep_dims is True:
                embeddings = torch.cat(all_embeddings, dim=0)
                return (embeddings, None)
                
            elif keep_dims == "flatten":
                embeddings = torch.cat(all_embeddings, dim=0)
                return (embeddings, all_doclens)
                
            else: 
                embeddings_list = []
                for batch_emb in all_embeddings:
                    for doc_emb in batch_emb:
                        embeddings_list.append(doc_emb)
                        
                return (embeddings_list, None)
        else:
            embeddings, doclens = self._encode_text_batch(docs, to_cpu)
            
            if keep_dims is True:
                return (embeddings, None)
            elif keep_dims == "flatten":
                embeddings = embeddings.unsqueeze(1)  # (document_num, 1, dim)
                return (embeddings, doclens)
            else: 
                embeddings_list = []
                for doc_emb in embeddings:
                    embeddings_list.append(doc_emb)
                return (embeddings_list, None)

    def imageFromPath(
        self, 
        image_paths,
        bsize=None,
        keep_dims=True,
        to_cpu=False, 
        showprogress=False,
        return_tokens=False, # not implemented
        pool_factor=1, # not implemented
        protected_tokens=0, # not implemented
        clustering_mode: str =  "hierarchical", # not implemented
    ):
        assert keep_dims in [True, False, "flatten"]
        # TODO: add clustering mode
        # assert clustering_mode in ["hierarchical"]
       
        if not self.has_image_encoder:
            raise ValueError("This model does not support image encoding")
            
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        bsize = bsize or self.config.hf_default_batch_size
        bsize = min(bsize, self.config.hf_max_batch_size)
        
        if bsize and len(image_paths) > bsize:
            all_embeddings = []
            iterator = range(0, len(image_paths), bsize)
            if showprogress:
                iterator = tqdm(iterator, desc="Encoding images", disable=self.verbose < 2)
                
            for i in iterator:
                batch_paths = image_paths[i:i + bsize]
                batch_embeddings = self._encode_image_batch(batch_paths, to_cpu)
                all_embeddings.append(batch_embeddings)
            embeddings = torch.cat(all_embeddings, dim=0)
            
            if keep_dims is True :
                return (embeddings, None)
            elif keep_dims == "flatten":
                doclens = [1]*embeddings.size(0)
                return (embeddings, doclens)
            else:  
                embeddings_list = [emb for emb in embeddings]
                return (embeddings_list, None)
        else:
            embeddings = self._encode_image_batch(image_paths, to_cpu)
            
            if keep_dims is True:
                return (embeddings, None)
            elif keep_dims == "flatten":
                embeddings = embeddings.unsqueeze(1)  # (document_num, 1, dim)
                doclens = [1] * embeddings.size(0)
                return (embeddings, doclens)
            else: 
                embeddings_list = [emb for emb in embeddings]
                return (embeddings_list, None)

    def imageFromPIL(self, images, bsize=None, keep_dims=True, to_cpu=False, showprogress=False):
        pass

    def _encode_text_inputs(self, inputs, to_cpu=False):
        if hasattr(self.model, 'get_text_features'):
            # CLIP 
            embeddings = self.model.get_text_features(**inputs)
        elif hasattr(self.model, 'get_text_embeddings'):
            # Some models have this method
            embeddings = self.model.get_text_embeddings(**inputs)
        else:
            # last hidden state
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token, (batch_size, dim)
            
        # Normalize if requested (actually always true)
        if self.config.hf_normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        if self.device.type == "cuda":
            embeddings = embeddings.half()
            
        return embeddings.cpu() if to_cpu else embeddings

    def _encode_text_batch(self, texts, to_cpu=False):
        inputs = self.processor(
            text=texts, 
            **self.config.hf_text_processor_config_
        )
        
        if self.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            with self.amp_manager.context():
                embeddings = self._encode_text_inputs(inputs)  # [batch_size, hidden_dim]
                doclens = [1] * embeddings.size(0)
                # if self.device.type == "cuda":
                #     embeddings.half()
                return embeddings, doclens

    def _encode_image_batch(self, image_paths, to_cpu=False):
        processed_images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path)
           
            if img.mode in ('P', 'LA', 'PA'):
                img = img.convert('RGBA').convert('RGB')
            else:
                img = img.convert('RGB')
            processed_images.append(img)
        
        inputs = self.processor(
            images=processed_images, 
            **self.config.hf_image_processor_config_
        )
    
        if self.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            with self.amp_manager.context():
                embeddings = self._encode_image_inputs(inputs) # (batch_size, hidden_dim)
                # if self.device.type == "cuda":
                #     embeddings.half()
                return embeddings.cpu() if to_cpu else embeddings

    def _encode_pil_batch(self, images, to_cpu=False):
        pass

    def _encode_image_inputs(self, inputs, to_cpu=False):
        """Encode image inputs and return embeddings."""
        if hasattr(self.model, 'get_image_features'):
            # CLIP
            embeddings = self.model.get_image_features(**inputs)
        elif hasattr(self.model, 'get_image_embeddings'):
            # Some models have this method
            embeddings = self.model.get_image_embeddings(**inputs)
        else:
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
         
        # Normalize if requested
        if self.config.hf_normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if self.device.type == "cuda":
            embeddings = embeddings.half()
        
        return embeddings.cpu() if to_cpu else embeddings
    
    def _stack_3D_tensors(groups):
        # bsize = sum([x.size(0) for x in groups])
        # maxlen = max([x.size(1) for x in groups])
        # hdim = groups[0].size(2)

        # output = torch.zeros(
        #     bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
        # )

        # offset = 0
        # for x in groups:
        #     endpos = offset + x.size(0)
        #     output[offset:endpos, : x.size(1)] = x
        #     offset = endpos

        # return output
        pass

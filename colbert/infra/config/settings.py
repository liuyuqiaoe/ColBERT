import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
    The defaults here have a special status in Run(), which initially calls assign_defaults(),
    so these aren't soft defaults in that specific context.
    """

    overwrite: bool = DefaultVal(False)

    root: str = DefaultVal(os.path.join(os.getcwd(), "experiments"))
    experiment: str = DefaultVal("default")

    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))

    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)

    avoid_fork_if_possible: bool = DefaultVal(False)

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(",")

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(
            device_idx in range(0, self.total_visible_gpus) for device_idx in value
        ), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, "indexes/")

    @property
    def script_name_(self):
        if "__file__" in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd) :]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath) :]
                except:
                    pass

            assert script_path.endswith(".py")
            script_name = script_path.replace("/", ".").strip(".")[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return "none"

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = DefaultVal("[unused0]")
    doc_token_id: str = DefaultVal("[unused1]")
    query_token: str = DefaultVal("[Q]")
    doc_token: str = DefaultVal("[D]")


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)


@dataclass
class DocSettings:
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(220)
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens: bool = DefaultVal(False)
    interaction: str = DefaultVal("colbert")


@dataclass
class TrainingSettings:
    similarity: str = DefaultVal("cosine")

    bsize: int = DefaultVal(32)

    accumsteps: int = DefaultVal(1)

    lr: float = DefaultVal(3e-06)

    maxsteps: int = DefaultVal(500_000)

    save_every: int = DefaultVal(None)

    resume: bool = DefaultVal(False)

    ## NEW:
    warmup: int = DefaultVal(None)

    warmup_bert: int = DefaultVal(None)

    relu: bool = DefaultVal(False)

    nway: int = DefaultVal(2)

    use_ib_negatives: bool = DefaultVal(False)

    reranker: bool = DefaultVal(False)

    distillation_alpha: float = DefaultVal(1.0)

    ignore_scores: bool = DefaultVal(False)

    model_name: str = DefaultVal(None)  # DefaultVal('bert-base-uncased')


@dataclass
class IndexingSettings:
    index_path: str = DefaultVal(None)

    index_bsize: int = DefaultVal(64)

    nbits: int = DefaultVal(1)

    kmeans_niters: int = DefaultVal(4)

    resume: bool = DefaultVal(False)

    pool_factor: int = DefaultVal(1)

    clustering_mode: str = DefaultVal("hierarchical")

    protected_tokens: int = DefaultVal(0)

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    ncells: int = DefaultVal(None)
    centroid_score_threshold: float = DefaultVal(None)
    ndocs: int = DefaultVal(None)
    load_index_with_mmap: bool = DefaultVal(False)


@dataclass
class HFSettings:

    # Model settings
    hf_model_name: str = DefaultVal("openai/clip-vit-base-patch16")
    hf_model_type: str = DefaultVal("clip")

    hf_amp: bool = DefaultVal(True)  

    hf_default_batch_size: int = DefaultVal(32)
    hf_max_batch_size: int = DefaultVal(128)

    # Text encoding settings
    hf_text_max_length: int = DefaultVal(77)
    hf_text_truncation: bool = DefaultVal(True)
    hf_text_padding: bool = DefaultVal(True)
    hf_text_return_tensors: str = DefaultVal("pt")

    # Image encoding settings
    hf_image_size: int = DefaultVal(224)
    hf_image_mean: tuple = DefaultVal((0.485, 0.456, 0.406))
    hf_image_std: tuple = DefaultVal((0.229, 0.224, 0.225))
    hf_image_return_tensors: str = DefaultVal("pt")

    # Embedding settings
    hf_normalize_embeddings: bool = DefaultVal(True) # always normalize embeddings
    hf_embedding_dimension: int = DefaultVal(None)  # TODO: Auto-detected from model

    # Fallback settings
    hf_fallback_to_clip: bool = DefaultVal(True)
    hf_fallback_model_name: str = DefaultVal("openai/clip-vit-base-patch16")

    hf_trust_remote_code: bool = DefaultVal(False)

    @property
    def hf_model_config_(self):
        return {
            "trust_remote_code": self.hf_trust_remote_code,
        }
    
    def hf_validate(self):
        # TODO: model paramaters validation
        pass
    def hf_config_initialization(self):
        pass
    @property
    def hf_text_processor_config_(self):
        return {
            "max_length": self.hf_text_max_length,
            "truncation": self.hf_text_truncation,
            "padding": self.hf_text_padding,
            "return_tensors": self.hf_text_return_tensors,
        }
    
    @property
    def hf_image_processor_config_(self):
        return {
            "size": self.hf_image_size,
            "mean": self.hf_image_mean,
            "std": self.hf_image_std,
            "return_tensors": self.hf_image_return_tensors,
        }
    
    def get_hf_model_info(self) -> dict:
        return {
            "model_name": self.hf_model_name,
            "model_type": self.hf_model_type,
            "embedding_dimension": self.hf_embedding_dimension,
            "normalize_embeddings": self.hf_normalize_embeddings,
            "text_max_length": self.hf_text_max_length,
            "image_size": self.hf_image_size,
            "default_batch_size": self.hf_default_batch_size,
            "hf_amp": self.hf_amp,
        }
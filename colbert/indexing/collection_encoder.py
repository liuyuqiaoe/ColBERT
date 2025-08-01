import torch

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch

from colbert.modeling.checkpoint import HFCheckpoint

class CollectionEncoder:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        return embs, doclens

class HFCollectionEncoder:
    def __init__(self, config, checkpoint: HFCheckpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu)
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)
        return embs, doclens
    
    def encode_images(self, image_paths):
        Run().print(f"#> Encoding {len(image_paths)} images..")

        if len(image_paths) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []
            for image_baches in batch(image_paths, self.config.index_bsize * 50):
                embs_, doclens_ = self.checkpoint.imageFromPath(
                    image_paths=image_baches,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    to_cpu= not self.use_gpu
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens


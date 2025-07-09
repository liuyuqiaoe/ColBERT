
# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import itertools

from colbert.evaluation.loaders import load_collection
from colbert.infra.run import Run


class Collection:
    def __init__(self, path=None, data=None):
        self.path = path
        self.data = data or self._load_file(path)

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.data.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.data[item]

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.data)

    def _load_file(self, path):
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        raise NotImplementedError()

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def save(self, new_path):
        assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)
            
            return f.name

    def enumerate(self, rank):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj
        
        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.

class ImageCollection(Collection):
    def __init__(self, image_paths=None, path=None):
        if image_paths is not None:
            self.data = [image_path for image_path in image_paths]
            self.path = None
        elif path is not None:
            self.data = self._load_image_paths(path)
            self.path = path
        else:
            raise ValueError("Must provide image_paths or path")

    def _load_image_paths(self, path):
        # Supports .txt or .csv with one image path per line
        with open(path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(image_paths)} image paths from {path}")
        return image_paths

    def _load_tsv(self, path):
        raise NotImplementedError("TSV loading is not supported for images.")

    def _load_jsonl(self, path):
        raise NotImplementedError("JSONL loading is not supported for images.")
    
    def _load_file(self, path):
        raise NotImplementedError("ImageCollection does not support loading from file.")

    def save(self, new_path):
        assert new_path.endswith('.txt'), "Only .txt saving is supported for images."
        assert not os.path.exists(new_path), new_path
        with open(new_path, 'w') as f:
            for p in self.data:
                f.write(f"{p}\n")
        return new_path
    
    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(image_paths=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

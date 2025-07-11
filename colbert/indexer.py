import os
import time

import torch.multiprocessing as mp

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import Launcher

from colbert.utils.utils import create_directory, print_message

from colbert.indexing.collection_indexer import encode
from colbert.indexing.image_collection_indexer import encode as image_encode
from colbert.modeling.checkpoint import HFCheckpoint

from typing import Optional

class Indexer:
    def __init__(self, checkpoint, config=None, verbose: int = 3):

        self.index_path = None
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)

        self.config = ColBERTConfig.from_existing(self.checkpoint_config, config, Run().config)
        self.configure(checkpoint=checkpoint)

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        return self.index_path

    def erase(self, force_silent: bool = False):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            if not force_silent:
                print_message(f"#> Will delete {len(deleted)} files already at {directory} in 20 seconds...")
                time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, collection, overwrite=False):
        assert overwrite in [True, False, 'reuse', 'resume', "force_silent_overwrite"]

        self.configure(collection=collection, index_name=name, resume=overwrite=='resume')
        # Note: The bsize value set here is ignored internally. Users are encouraged
        # to supply their own batch size for indexing by using the index_bsize parameter in the ColBERTConfig.
        self.configure(bsize=64, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = (not os.path.exists(self.config.index_path_))

        assert (overwrite in [True, 'reuse', 'resume', "force_silent_overwrite"]) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite == 'force_silent_overwrite':
            self.erase(force_silent=True)
        elif overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            self.__launch(collection)

        return self.index_path

    def __launch(self, collection):
        launcher = Launcher(encode)
        if self.config.nranks == 1 and self.config.avoid_fork_if_possible:
            shared_queues = []
            shared_lists = []
            launcher.launch_without_fork(self.config, collection, shared_lists, shared_queues, self.verbose)

            return

        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        # Encodes collection into index using the CollectionIndexer class
        launcher.launch(self.config, collection, shared_lists, shared_queues, self.verbose)

class ImageIndexer:
    def __init__(self, config=None, checkpoint: HFCheckpoint = None, verbose: int = 3):
        self.index_path = None
        self.verbose = verbose
        # TODO: pre-defined hfcheckpoint
        self.checkpoint = checkpoint
        # self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)

        # self.config = ColBERTConfig.from_existing(config, Run().config)
        # self.configure(checkpoint=checkpoint)
        # For images, we don't need a checkpoint since we use CLIP
        self.config = ColBERTConfig.from_existing(config, Run().config)
        # TODO: clip configure
        # self.configure()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        return self.index_path

    def erase(self, force_silent: bool = False):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            if not force_silent:
                print_message(f"#> Will delete {len(deleted)} files already at {directory} in 20 seconds...")
                time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def __launch(self, collection):
            launcher = Launcher(image_encode)  # Use image_encode instead of encode
            
            if self.config.nranks == 1 and self.config.avoid_fork_if_possible:
                shared_queues = []
                shared_lists = []
                launcher.launch_without_fork(self.config, collection, shared_lists, shared_queues, self.verbose)
                return

            manager = mp.Manager()
            shared_lists = [manager.list() for _ in range(self.config.nranks)]
            shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

            # Encode collection into index using the ImageCollectionIndexer class
            launcher.launch(self.config, collection, shared_lists, shared_queues, self.verbose)

    def index(self, name, collection, overwrite=False):
            assert overwrite in [True, False, 'reuse', 'resume', "force_silent_overwrite"]

            self.configure(collection=collection, index_name=name, resume=overwrite=='resume')
            # TODO: Adjust the batch size for images
            self.configure(bsize=32, index_bsize=16, partitions=None)

            self.index_path = self.config.index_path_
            index_does_not_exist = (not os.path.exists(self.config.index_path_))

            assert (overwrite in [True, 'reuse', 'resume', "force_silent_overwrite"]) or index_does_not_exist, self.config.index_path_
            create_directory(self.config.index_path_)

            if overwrite == 'force_silent_overwrite':
                self.erase(force_silent=True)
            elif overwrite is True:
                self.erase()

            if index_does_not_exist or overwrite != 'reuse':
                self.__launch(collection)

            return self.index_path   
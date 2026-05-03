"""Stream tokens for pre-training"""

from collections.abc import Generator

import torch
from datasets.iterable_dataset import IterableDataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class TokenStream:
    def __init__(
        self,
        stream_dataset: IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 2048,
        doc_buffer_size: int = 8,
    ):
        self.dataset = stream_dataset
        self.tokenizer = tokenizer
        self.EOS_TOKEN_ID = tokenizer.eos_token_id
        self.doc_buffer_size = doc_buffer_size
        self.max_seq_len = max_seq_len

    def _document_stream(self) -> Generator[list[str], None, None]:
        """Yields batches of documents as lists of strings"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = self.dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )
        else:
            ds = self.dataset

        batch = []
        for example in ds:
            batch.append(example["text"])
            if len(batch) == self.doc_buffer_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __iter__(self) -> Generator[dict[str, Tensor], None, None]:
        """Yields batches of token ids as dicts with key 'input_ids'"""
        token_buffer = []
        for document_batch in self._document_stream():
            token_batch = self.tokenizer(
                document_batch, truncation=False, add_special_tokens=False
            ).input_ids

            for tokens in token_batch:
                token_buffer.extend([self.EOS_TOKEN_ID])
                token_buffer.extend(tokens)

                while len(token_buffer) >= self.max_seq_len:
                    chunk = torch.tensor(token_buffer[: self.max_seq_len])
                    token_buffer = token_buffer[self.max_seq_len :]
                    yield {"input_ids": chunk}

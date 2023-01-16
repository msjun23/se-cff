from pathlib import Path

import torch
import torch.utils.data

from .sequence import Sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=15, representation='voxel'):
        self.dataset_path = dataset_path
        self.delta_t_ms   = delta_t_ms
        self.num_bins     = num_bins
        self.representation = representation

    def get_train_dataset(self):
        train_path = self.dataset_path / 'train'
        assert self.dataset_path.is_dir(), str(self.dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        for child in train_path.iterdir():
            train_sequences.append(Sequence(child, 'train', self.delta_t_ms, self.num_bins, self.representation))

        train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        
        return train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

    def get_test_dataset(self):
        test_path = self.dataset_path / 'test'
        assert self.dataset_path.is_dir(), str(self.dataset_path)
        assert test_path.is_dir(), str(test_path)

        test_sequences = list()
        for child in test_path.iterdir():
            test_sequences.append(Sequence(child, 'test', self.delta_t_ms, self.num_bins, self.representation))

        test_dataset = torch.utils.data.ConcatDataset(test_sequences)
        
        return test_dataset

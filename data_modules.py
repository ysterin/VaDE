import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, FashionMNIST
import pytorch_lightning as pl
import numpy as np


class TripletDataset(IterableDataset):
    def __init__(self, data, labels, transform=None, data_size=None, max_samples=None, seed=None):
        super(TripletDataset, self).__init__()
        assert len(data) == len(labels)
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed=self.seed))
        if data_size and data_size < len(data):
            idxs = self.rng.choice(len(data), size=data_size, replace=False)
            data, labels = data[idxs], labels[idxs]
        self.data_size = len(data)
        self.data = data
        self.labels = labels
        self.label_set = list(set(labels.numpy()))
        self.data_dict = {lbl: [self.data[i] for i in range(self.data_size) if self.labels[i]==lbl] \
                            for lbl in self.label_set}
        self.n_classes = len(self.label_set)
        self.class_sizes = {lbl: len(self.data_dict[lbl]) for lbl in self.label_set}
        if not max_samples:
            max_samples = sum([n*(n-1)//2 * (self.data_size-n) for n in self.class_sizes.values()])
        self.max_samples = max_samples


    def __len__(self):
        return self.max_samples        

    def __iter__(self):
        self.rng = np.random.Generator(np.random.PCG64(seed=self.seed))
        for i in range(self.max_samples):
            anchor_label, neg_label = self.rng.choice(self.label_set, size=2, replace=False)
            try:
                anchor_idx, positive_idx = self.rng.choice(self.class_sizes[anchor_label], size=2, replace=False)
            except ValueError as e:
                continue
            negative_idx = self.rng.choice(self.class_sizes[neg_label], size=1)[0]
            yield {'anchor': self.data_dict[anchor_label][anchor_idx], 
                   'positive': self.data_dict[anchor_label][positive_idx], 
                   'negative': self.data_dict[neg_label][negative_idx]}


class CombinedDataset(Dataset):
    def __init__(self, dataset, transform=None, data_size=None, max_triplets=None, seed=42):
        super(CombinedDataset, self).__init__()
        self.data = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
        targets = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
        if not max_triplets:
            max_triplets = len(dataset)
        self.triplet_dataset = TripletDataset(self.data, targets, transform, data_size, max_samples=max_triplets, seed=seed)
        self.triplets_iterator = iter(self.triplet_dataset)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i, (sample, triplet) in enumerate(zip(self.data ,self.triplet_dataset)):
            yield (sample, triplet)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            triplet = next(self.triplets_iterator)
        except StopIteration:
            self.triplets_iterator = iter(self.triplet_dataset)
            triplet = next(self.triplets_iterator)
        return (sample, triplet)


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, valid_ds, batch_size=256, num_workers=1):
        super(BasicDataModule, self).__init__()
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_ds, self.valid_ds = train_ds, valid_ds
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, path='data', bs=256, dataset='mnist', data_size=None, seed=42):
        super(MNISTDataModule, self).__init__()
        self.path = path
        self.batch_size = bs
        self.dataset = dataset
        self.data_size = data_size
        self.seed = seed
    
    def prepare_data(self, stage_name=None):
        self.train_ds = MNIST(self.path, download=True, train=True)
        self.valid_ds = MNIST(self.path, download=True, train=False)
        if self.dataset == 'mnist':
            self.train_ds = MNIST("data", download=True)
            self.valid_ds = MNIST("data", download=True, train=False)
        elif self.dataset == 'fmnist':
            self.train_ds = FashionMNIST("data", download=True)
            self.valid_ds = FashionMNIST("data", download=True, train=False)
        def to_tensor_dataset(ds):
            X = ds.data.view(-1, 28**2).float()/255.
            return TensorDataset(X, ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [self.train_ds, self.valid_ds])
        if self.data_size is not None:
            # n_sample = self.data_size
            # to_subset = lambda ds: torch.utils.data.random_split(ds, 
            #                                                      [n_sample, len(ds) - n_sample],
            #                                                      torch.Generator().manual_seed(42))[0]
            # self.train_ds = to_subset(self.train_ds)
            self.train_ds = torch.utils.data.random_split(self.train_ds, 
                                                          [self.data_size, len(self.train_ds) - self.data_size],
                                                          torch.Generator().manual_seed(self.seed))[0]
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])
                
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=1, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=1)


def get_data_and_targets(dataset):
    data = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    targets = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
    return data, targets


class TripletsDataModule(pl.LightningDataModule):
    def __init__(self, base_datamodule, n_samples_for_triplets=None, n_triplets=None, n_triplets_valid=None, batch_size=256):
        super(TripletsDataModule, self).__init__()
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.n_samples_for_triplets = n_samples_for_triplets
        self.n_triplets = n_triplets
        if n_triplets_valid is None:
            n_triplets_valid = n_triplets
        self.n_triplets_valid = n_triplets_valid
    
    def prepare_data(self, stage_name=None):
        self.base_datamodule.prepare_data()
        self.train_ds = TripletDataset(*get_data_and_targets(self.base_datamodule.train_ds), data_size=self.n_samples_for_triplets, 
                                                max_samples=self.n_triplets)
        self.valid_ds = TripletDataset(*get_data_and_targets(self.base_datamodule.valid_ds), data_size=self.n_samples_for_triplets,
                                                max_samples=self.n_triplets_valid)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=0)


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, base_datamodule, n_samples_for_triplets=None, n_triplets=None, 
                 n_triplets_valid=None, batch_size=256, seed=42):
        super(CombinedDataModule, self).__init__()
        self.base_datamodule = base_datamodule
        self.batch_size, self.seed = batch_size, seed
        self.n_samples_for_triplets = n_samples_for_triplets
        self.n_triplets = n_triplets
        if n_triplets_valid is None:
            n_triplets_valid = n_triplets
        self.n_triplets_valid = n_triplets_valid
    
    def prepare_data(self):
        self.base_datamodule.prepare_data()
        self.train_ds = CombinedDataset(self.base_datamodule.train_ds, data_size=self.n_samples_for_triplets,
                                        max_triplets=self.n_triplets, seed=self.seed)
        self.valid_ds = CombinedDataset(self.base_datamodule.valid_ds, data_size=self.n_samples_for_triplets,
                                        max_triplets=self.n_triplets_valid, seed=self.seed)
        # self.train_ds = TripletDataset(*get_data_and_targets(self.base_datamodule.train_ds), data_size=self.n_samples_for_triplets, 
        #                                         max_samples=self.n_triplets)
        # self.valid_ds = TripletDataset(*get_data_and_targets(self.base_datamodule.valid_ds), data_size=self.n_samples_for_triplets,
        #                                         max_samples=self.n_triplets_valid)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=0)
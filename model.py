"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import gzip
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import torch


def load_fashion_mnist(n_train: int = 10000, n_test: int = 2000):
    '''download the four idx gz files once, parse with np.frombuffer, return float tensors in 0-1 and int64 labels.'''

    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    filenames = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]

    temp_dir = Path(tempfile.gettempdir())

    paths = dict()

    for filename in filenames:
        path = temp_dir / filename
        paths[filename] = path

        if not path.exists():
            url = urljoin(base_url, filename)
            urllib.request.urlretrieve(url, path)

    def parse_images(path: Path):
        with gzip.open(path, 'rb') as f:
            data = f.read()

        images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28)

        return images

    def parse_labels(path: Path):
        with gzip.open(path, 'rb') as f:
            data = f.read()

        labels = np.frombuffer(data, dtype=np.uint8, offset=8)

        return labels

    train_images = parse_images(paths['train-images-idx3-ubyte.gz'])
    train_labels = parse_labels(paths['train-labels-idx1-ubyte.gz'])

    test_images = parse_images(paths['t10k-images-idx3-ubyte.gz'])
    test_labels = parse_labels(paths['t10k-labels-idx1-ubyte.gz'])

    X_train = torch.tensor(train_images[:n_train], dtype=torch.float32) / 255
    y_train = torch.tensor(train_labels[:n_train], dtype=torch.int64)
    X_test = torch.tensor(test_images[:n_test], dtype=torch.float32) / 255
    y_test = torch.tensor(test_labels[:n_test], dtype=torch.int64)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

# Step 2 - FashionDataset
from torch import Tensor
from torch.utils.data import Dataset


class FashionDataset(Dataset):

    def __init__(self, X: Tensor, y: Tensor, mean: float = 0.2860, std: float = 0.3530):
        '''store X, y, mean, std'''

        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        '''((X[i] - mean) / std, y[i])'''

        return (self.X[i] - self.mean) / self.std, self.y[i]

# Step 3 - make_loaders
from torch.utils.data import DataLoader


def make_loaders(
    data: dict[str, Tensor], batch_size: int = 64, val_size: int = 2000, seed: int = 42
):
    '''last val_size training images -> validation; seeded shuffled train loader; return loaders + sizes.'''

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    train_dataset = FashionDataset(X_train, y_train)
    val_dataset = FashionDataset(X_val, y_val)
    test_dataset = FashionDataset(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'sizes': (len(train_dataset), len(val_dataset), len(test_dataset)),
    }

# Step 4 - MLP
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, hidden1: int = 300, hidden2: int = 100, n_classes: int = 10):
        super().__init__()
        '''fc1 (784 -> hidden1), fc2 (hidden1 -> hidden2), out (hidden2 -> n_classes)'''

        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        '''flatten, ReLU after fc1 and fc2, return logits'''

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)


def count_parameters(model: nn.Module):
    '''number of trainable parameters as an int'''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 5 - train_one_epoch
from typing import Callable

from torch.optim import Optimizer


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
):
    '''model.train(); per batch: zero_grad, forward, loss, backward, step; return mean batch loss.'''

    model.train()
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

# Step 6 - evaluate
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: Callable[[Tensor, Tensor], Tensor]):
    '''eval mode + no_grad; return (mean loss over examples, accuracy) as floats.'''

    model.eval()
    total_loss = 0
    correct: int = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n

# Step 7 - fit
import copy


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    epochs: int = 5,
    lr: float = 0.05,
    seed: int = 42,
):
    '''seeded SGD training with a validation pass per epoch; restore the best-val-accuracy weights; return history.'''

    torch.manual_seed(seed)

    train_loader = loaders['train']
    val_loader = loaders['val']

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loss = []
    val_loss = []
    val_acc = []
    best_acc = -1
    for epoch in range(epochs):
        epoch_train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, loss_fn)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return {
        'best_epoch': best_epoch,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss,
    }

# Step 8 - lr_range_test
from typing import Type


def lr_range_test(
    make_model: Type[nn.Module],
    loader: DataLoader,
    lrs: list[float],
    n_batches: int = 20,
    seed: int = 42,
):
    '''fresh seeded model per lr; mean loss over the first n_batches; return {lr: loss}.'''

    loss_fn = nn.CrossEntropyLoss()

    lr2loss = dict()
    for lr in lrs:
        torch.manual_seed(seed)

        model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr)

        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss: Tensor = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if len(losses) >= n_batches:
                break

        lr2loss[lr] = sum(losses) / len(losses)

    return lr2loss

# Step 9 - random_search
def random_search(
    loaders: dict[str, DataLoader], n_trials: int = 4, epochs: int = 2, seed: int = 42
):
    '''seeded random configurations of (hidden1, hidden2, lr); fit each; return trials and the best.'''

    rng = np.random.default_rng(seed)

    trials = []
    best_acc = -1
    for _ in range(n_trials):

        hidden1: int = rng.choice([100, 200, 300])
        hidden2: int = rng.choice([50, 100])
        lr: float = rng.choice([0.01, 0.05, 0.1])

        torch.manual_seed(seed)
        model = MLP(hidden1, hidden2)
        history = fit(model, loaders, epochs, lr, seed)
        val_acc: float = max(history['val_acc'])

        trail = {
            'hidden1': hidden1,
            'hidden2': hidden2,
            'lr': lr,
            'val_acc': val_acc,
        }
        trials.append(trail)
        if val_acc > best_acc:
            best_acc = val_acc
            best = trail

    return {
        'trials': trials,
        'best': best,
    }

# Step 10 - test_accuracy
def test_accuracy(model: nn.Module, loaders: dict[str, DataLoader]):
    '''accuracy on loaders['test'] via evaluate.'''

    test_loader = loaders['test']

    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits: Tensor = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return correct / n

# Step 11 - save_model
def save_model(model: MLP, path: str):
    '''torch.save({'state_dict': ..., 'config': {'hidden1', 'hidden2', 'n_classes'}}, path)'''

    torch.save(
        {
            'state_dict': model.state_dict(),
            'config': {
                'hidden1': model.fc1.bias.shape[0],
                'hidden2': model.fc2.bias.shape[0],
                'n_classes': model.out.bias.shape[0],
            },
        },
        path,
    )


def load_model(path: str):
    '''rebuild MLP from the config, load the state dict, eval(), return it.'''

    p = torch.load(path)
    state_dict = p['state_dict']
    config = p['config']
    hidden1 = config['hidden1']
    hidden2 = config['hidden2']
    n_classes = config['n_classes']

    model = MLP(hidden1, hidden2, n_classes)
    model.load_state_dict(state_dict)
    model.eval()

    return model

# Step 12 - predict_classes (not yet solved)
# TODO: implement


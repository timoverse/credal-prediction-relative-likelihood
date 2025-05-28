import numpy as np
import probly.data
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import pickle
from probly.data import DCICDataset


CIFAR_SIZE = 32
PATH = '/home/scratch/likelihood-ensembles/datasets'
NUM = 10000
NUM_WORKERS = 4
VAL_SPLIT = 0.2

def get_data_train(name, seed, batch_size=128, validation=True):
    if name == 'cifar10':
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True,
                                             transform=transform_train)
        test = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True,
                                            transform=transform_test)
    elif name == 'qualitymri':
        transforms = T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize((0.1485, 0.1485, 0.1485), (0.1819, 0.1819, 0.1819))])
        train = DCICDataset(root=f'{PATH}/QualityMRI', transform=transforms, first_order=False)
        train, test = torch.utils.data.random_split(train, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
    elif name == 'chaosnli':
        with open(f'{PATH}/embeddings/snli.pkl', 'rb') as f:
            snli = pickle.load(f)

        with open(f'{PATH}/embeddings/mnli_m.pkl', 'rb') as f:
            mnli = pickle.load(f)

        embedding = np.concatenate((snli["embedding"], mnli["embedding"]), axis=0)
        premise = np.concatenate((snli["premise"], mnli["premise"]), axis=0)
        hypothesis = np.concatenate((snli["hypothesis"], mnli["hypothesis"]), axis=0)
        label_dist = np.concatenate((snli["label_dist"], mnli["label_dist"]), axis=0)

        X_train, X_test, lambda_train, lambda_test, premise_train, premise_test, hypothesis_train, hypothesis_test = train_test_split(
            embedding, label_dist, premise, hypothesis, test_size=0.2, random_state=seed)
        # sample randomly from the ground truth first order distribution
        y_train = np.array([np.random.choice([0, 1, 2], p=lambda_train[i]) for i in range(len(lambda_train))])
        y_test = np.array([np.random.choice([0, 1, 2], p=lambda_test[i]) for i in range(len(lambda_test))])
        train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    else:
        raise ValueError(f"Dataset {name} not found")

    if validation:
        train, val = torch.utils.data.random_split(train, [1 - VAL_SPLIT, VAL_SPLIT],
                                                   generator=torch.Generator().manual_seed(seed))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
                                                 num_workers=NUM_WORKERS)
    else:
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = None
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader


def get_data_task(name, seed, first_order=True, batch_size=128):
    if name == 'cifar10' or name == 'cifar10h':
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if first_order:
            test = probly.data.CIFAR10H(root=PATH, transform=transforms)
        else:
            test = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True,
                                            transform=transforms)
    elif name == 'chaosnli':
        with open(f'{PATH}/embeddings/snli.pkl', 'rb') as f:
            snli = pickle.load(f)

        with open(f'{PATH}/embeddings/mnli_m.pkl', 'rb') as f:
            mnli = pickle.load(f)

        embedding = np.concatenate((snli["embedding"], mnli["embedding"]), axis=0)
        premise = np.concatenate((snli["premise"], mnli["premise"]), axis=0)
        hypothesis = np.concatenate((snli["hypothesis"], mnli["hypothesis"]), axis=0)
        label_dist = np.concatenate((snli["label_dist"], mnli["label_dist"]), axis=0)

        X_train, X_test, lambda_train, lambda_test, premise_train, premise_test, hypothesis_train, hypothesis_test = train_test_split(
            embedding, label_dist, premise, hypothesis, test_size=0.2, random_state=seed)
        if first_order:
            test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(lambda_test))
        else:
            # sample randomly from the ground truth first order distribution
            y_test = np.array([np.random.choice([0, 1, 2], p=lambda_test[i]) for i in range(len(lambda_test))])
            test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    elif name == 'qualitymri':
        transforms = T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize((0.1485, 0.1485, 0.1485), (0.1819, 0.1819, 0.1819))])
        train = DCICDataset(root=f'{PATH}/QualityMRI', transform=transforms, first_order=first_order)
        _, test = torch.utils.data.random_split(train, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
    else:
        raise ValueError(f"Dataset {name} not found")

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    return test_loader


def get_data_ood(name_id, name_ood, seed, batch_size=128):
    if name_id == 'cifar10':
        transforms = T.Compose([T.Resize((CIFAR_SIZE,CIFAR_SIZE)),
                                T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])


        test_id = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True,
                                               transform=transforms)
    else:
        raise ValueError(f"Dataset {name_id} not found")

    if name_ood == 'svhn':
        test_ood = torchvision.datasets.SVHN(root=PATH, split='test', download=True,
                                             transform=transforms)
    elif name_ood == 'cifar100':
        test_ood = torchvision.datasets.CIFAR100(root=PATH, train=False, download=True,
                                                 transform=transforms)
    elif name_ood == 'places365':
        test_ood = torchvision.datasets.Places365(root=PATH, split="val", download=True,
                                                  small=True, transform=transforms)
    elif name_ood == 'fmnist':
        transforms = T.Compose([
            T.Grayscale(3),
            *transforms.transforms
        ])
        test_ood = torchvision.datasets.FashionMNIST(root=PATH, train=False, download=True, transform=transforms)
    elif name_ood == 'imagenet':
        test_ood = torchvision.datasets.ImageNet(root=PATH + "/imagenet/", split="val", transform=transforms)
    else:
        raise ValueError(f"Dataset {name_ood} not found")

    rng = torch.Generator().manual_seed(seed)
    random_id = torch.utils.data.RandomSampler(test_id, num_samples=NUM, generator=rng)
    random_ood = torch.utils.data.RandomSampler(test_ood, num_samples=NUM, generator=rng)
    id_loader = torch.utils.data.DataLoader(test_id, batch_size=batch_size, sampler=random_id,
                                            num_workers=NUM_WORKERS)
    ood_loader = torch.utils.data.DataLoader(test_ood, batch_size=batch_size, sampler=random_ood,
                                             num_workers=NUM_WORKERS)

    return id_loader, ood_loader
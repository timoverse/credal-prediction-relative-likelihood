import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from data import get_data_train
import models
import utils
import json
import argparse
from models import DesterckeEnsemble, CaprioEnsemble, WangEnsemble
from probly.losses import ELBOLoss
from probly.representation import Ensemble

PATH = '/home/scratch/likelihood-ensembles/new_checkpoints'


def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_credal_wrapper(ensemble, train_loader, Optimizer, device, lr, wd, epochs, val_loader=None):
    criterion = nn.CrossEntropyLoss()
    for model in tqdm(ensemble.models):
        if isinstance(Optimizer, optim.SGD):
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
        if isinstance(optimizer, optim.SGD):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if isinstance(optimizer, optim.SGD):
                scheduler.step()
        if val_loader is not None:
            model.eval()
            val_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            print(f"Validation loss: {val_loss / len(val_loader)}")

def train_credal_ensemble(ensemble, train_loader, Optimizer, device, lr, wd, epochs, val_loader=None):
    criterion = nn.CrossEntropyLoss()
    for model in tqdm(ensemble.models):
        if isinstance(Optimizer, optim.SGD):
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
        if isinstance(optimizer, optim.SGD):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if isinstance(optimizer, optim.SGD):
                scheduler.step()
        if val_loader is not None:
            model.eval()
            val_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            print(f"Validation loss: {val_loss / len(val_loader)}")

def train_credal_bayesian_deep_learning(ensemble, train_loader, Optimizer, device, lr, wd, epochs, val_loader=None):
    criterion = ELBOLoss(1e-7)
    for model in tqdm(ensemble.models):
        if Optimizer == optim.SGD:
            print("Just to be sure, using SGD with momentum")
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
        if isinstance(optimizer, optim.SGD):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets, model.kl_divergence)
                loss.backward()
                optimizer.step()
            if isinstance(optimizer, optim.SGD):
                scheduler.step()
        if val_loader is not None:
            model.eval()
            val_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, model.kl_divergence)
                    val_loss += loss.item()
            print(f"Validation loss: {val_loss / len(val_loader)}")


def train_credal_net_ensembles(ensemble, train_loader, Optimizer, device, lr, wd, epochs=25, val_loader=None):
    """
    https://proceedings.neurips.cc/paper_files/paper/2024/file/911fc798523e7d4c2e9587129fcf88fc-Paper-Conference.pdf
    """
    no_red_criterion = nn.NLLLoss(reduction='none')
    criterion = nn.NLLLoss()
    for model in tqdm(ensemble.models):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if isinstance(Optimizer, optim.SGD):
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
        if isinstance(optimizer, optim.SGD):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for e in range(epochs):
            model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                outputs_up = outputs[:, ensemble.n_classes:]
                loss_up = criterion(torch.log(outputs_up), targets)

                outputs_lo = outputs[:, :ensemble.n_classes]
                loss_lo = no_red_criterion(torch.log(outputs_lo), targets)

                # Select top delta * batch_size samples with highest loss for backward
                loss_lo_sort, _ = torch.sort(loss_lo, descending=True, dim=-1)

                bound_index = int(np.floor(ensemble.delta * targets.shape[0])) - 1
                bound_value = loss_lo_sort[bound_index]

                choose_index = torch.greater_equal(loss_lo, bound_value)
                choose_outputs_lo = outputs_lo[choose_index]
                choose_targets = targets[choose_index]

                loss_lo_mod = criterion(torch.log(choose_outputs_lo), choose_targets)
                loss = loss_lo_mod + loss_up
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            if isinstance(optimizer, optim.SGD):
                scheduler.step()
            if e%10 == 0:
                print(f"Epoch: {e}, Loss: {total_loss/len(train_loader.dataset)}")


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        device = utils.get_best_gpu()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = 'cpu'

    print(f'Using device: {device} and the following args:')
    utils.printargs(args)

    train_loader, val_loader, _ = get_data_train(args.dataset, args.seed, validation=args.validate)

    model = models.get_model(args.model, args.classes)

    if args.baseline == "credalwrapper":
        ensemble = Ensemble(model, args.n_members)
        ensemble = ensemble.to(device)
        train_credal_wrapper(ensemble, train_loader, optim.Adam if args.optimizer == "adam" else optim.SGD, device, args.lr, args.wd, args.epochs, val_loader=val_loader if args.validate else None)
    elif args.baseline == "credalensembling":
        ensemble = DesterckeEnsemble(model, args.n_members)
        ensemble = ensemble.to(device)
        train_credal_wrapper(ensemble, train_loader, optim.Adam if args.optimizer == "adam" else optim.SGD, device, args.lr, args.wd, args.epochs,
                             val_loader=val_loader if args.validate else None)
    elif args.baseline == "credalbnn":
        ensemble = CaprioEnsemble(model, args.n_members, args.prior_mu, args.prior_std)
        ensemble = ensemble.to(device)
        train_credal_bayesian_deep_learning(ensemble, train_loader, optim.Adam if args.optimizer == "adam" else optim.SGD, device, args.lr, args.wd, args.epochs,
                             val_loader=val_loader if args.validate else None)
    elif args.baseline == "credalnet":
        ensemble = WangEnsemble(model, args.n_members, args.delta, args.classes)
        ensemble = ensemble.to(device)
        train_credal_net_ensembles(ensemble, train_loader, optim.Adam if args.optimizer == "adam" else optim.SGD, device, args.lr, args.wd, args.epochs,
                             val_loader=val_loader if args.validate else None)

        #torch.save(ensemble.state_dict(), f'{PATH}/baseline_{args.dataset}_{args.baseline}_{args.n_members}_{args.delta}_{args.seed}')
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    torch.save(ensemble.state_dict(), f'{PATH}/baseline_{args.dataset}_{args.baseline}_{args.n_members}_{args.seed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, help='Baseline')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--validate', type=_str2bool, help='Use validation set (20%) or not?')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--n_members', type=int, help='Number of members to use')
    parser.add_argument('--optimizer', type=str, help='The optimizer to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--prior_mu', type=json.loads, help='Prior mean')
    parser.add_argument('--prior_std', type=json.loads, help='Prior std')
    parser.add_argument('--delta', type=float, help='Delta parameter')
    args = parser.parse_args()
    main(args)

# python train_baseline.py --baseline=credalnet --dataset=chaosnli --validate=False --classes=3 --model=crefcnet --n_members=20 --optimizer=adam --epochs=200 --lr=0.01 --wd=0 --seed=1 --delta=0.5
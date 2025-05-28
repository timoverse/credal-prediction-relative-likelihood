import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from data import get_data_train
import models
import utils
import argparse
from models import LikelihoodEnsemble, get_model
import os

PATH = '/home/scratch/likelihood-ensembles/checkpoints'

os.environ["TORCH_HOME"] = PATH

def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_mle(mle, train_loader, device, Optimizer, lr, wd, epochs, val_loader=None):
    mle.to(device)
    criterion = nn.CrossEntropyLoss()
    if isinstance(Optimizer, optim.SGD):
        optimizer = Optimizer(mle.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = Optimizer(mle.parameters(), lr=lr, weight_decay=wd)
    if isinstance(optimizer, optim.SGD):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        mle.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = mle(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch%50 == 0:
            print(f'Epoch {epoch+1}, loss {running_loss/len(train_loader)}')
        if isinstance(optimizer, optim.SGD):
            scheduler.step()

        if val_loader:
            mle.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = mle(inputs)
                    val_loss += criterion(outputs, targets).item()
                    correct += (outputs.argmax(1) == targets).sum().item()
                    total += targets.size(0)
            acc = 100 * correct / total
            val_loss /= len(val_loader)
            print(f'Loss {running_loss/len(train_loader)}, Validation loss: {val_loss}, accuracy: {acc}')


def train_ensemble(ensemble, alpha, train_loader, device, Optimizer, lr, wd, epochs, batch_check=True):
    if batch_check:
        X_train, y_train = utils.loader_to_tensor(train_loader)
        X_train = X_train.to(device)
        with torch.no_grad():
            outputs = ensemble.models[0](X_train)
        mll = utils.log_likelihood(outputs, y_train)
    else:
        mll = 0.0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = ensemble.models[0](inputs)
                mll += utils.log_likelihood_sum(outputs, targets)
            mll /= len(train_loader.dataset)

    print(f'MLE log likelihood: {mll}')

    ensemble.to(device)
    num_models = len(ensemble.models) - 1
    thresholds = torch.tensor(np.linspace(alpha, 1.0, num_models, endpoint=False, dtype=np.float32)).to(device)
    for i in tqdm(range(num_models)):
        model = ensemble.models[i+1].to(device)
        criterion = nn.CrossEntropyLoss()
        if isinstance(Optimizer, optim.SGD):
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
        if isinstance(optimizer, optim.SGD):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        reached = False
        last_epoch = None
        final_rl = None
        for epoch in range(epochs):
            last_epoch = epoch
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_check:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X_train)
                    ll = utils.log_likelihood(outputs, y_train)
                    rl = torch.exp(ll - mll)
                    final_rl = rl
                    if rl >= thresholds[i]:
                        reached = True
                        break
            if reached:
                break
            if isinstance(optimizer, optim.SGD):
                scheduler.step()
            if not batch_check:
                ll = 0
                model.eval()
                with torch.no_grad():
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        ll += utils.log_likelihood_sum(outputs, targets)
                    ll /= len(train_loader.dataset)
                rl = torch.exp(ll - mll)
                final_rl = rl
                if rl >= thresholds[i]:
                    reached = True
                    break
        ensemble.rls.append(final_rl.item())
        print(f'Reached threshold {thresholds[i]} {reached} after epoch {last_epoch} rl: {final_rl}')


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

    model = models.get_model(args.model, args.classes)

    train_loader, val_loader, _ = get_data_train(args.dataset, args.seed, validation=args.validate)

    if args.alpha is None:
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    else:
        alphas = [0.0, args.alpha]

    for i, alpha in enumerate(alphas):
        ensemble = LikelihoodEnsemble(model, args.classes, n_members=args.n_members, tobias_value=None if args.tobias == -1 else args.tobias)
        if i == 0:
            train_mle(ensemble.models[0], train_loader, device, optim.Adam if args.optimizer == "adam" else optim.SGD, args.lr, args.wd, args.epochs_mle if args.epochs_mle else args.epochs, val_loader=val_loader)
            mle = ensemble.models[0]
        else:
            ensemble.models[0] = mle
            train_ensemble(ensemble, alpha, train_loader, device, optim.Adam if args.optimizer == "adam" else optim.SGD, args.lr, args.wd, args.epochs, batch_check=args.batch)
        utils.save_ensemble(ensemble, f'{PATH}/{args.dataset}_{args.model}_{args.n_members}_{args.tobias}_{alpha}_{args.seed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--validate', type=_str2bool, help='Use validation set (20%) or not?')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--alpha', type=float, help='Alpha value to use')
    parser.add_argument('--n_members', type=int, help='Number of members to use')
    parser.add_argument('--optimizer', type=str, help='The optimizer to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--tobias', type=int, help='Tobias <3 value')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--batch', type=_str2bool, help='Check every batch or every epoch. True is every batch')
    parser.add_argument('--epochs_mle', type=int, help='Number of epochs to train MLE')
    args = parser.parse_args()
    main(args)

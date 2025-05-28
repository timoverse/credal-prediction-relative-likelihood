'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import sys
import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def intersection_probability(preds):
    mins = np.min(preds, axis=1)
    maxs = np.max(preds, axis=1)
    diffs = maxs - mins
    alphas = (1 - np.sum(mins, axis=1)) / (np.sum(diffs, axis=1))
    int_preds = mins + alphas[:, None] * diffs
    return int_preds


def log_likelihood(outputs, targets):
    # outputs should be logits
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.mean(outputs[torch.arange(outputs.shape[0]), targets])
    return ll


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
term_width = int(80)

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('../credal-prediction-relative-likelihood1')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def loader_to_tensor(loader):
    x = []
    y = []
    for data in loader:
        x.append(data[0])
        y.append(data[1])
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y

def log_likelihood_sum(outputs, targets):
    # outputs should be logits
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.sum(outputs[torch.arange(outputs.shape[0]), targets.int()])
    return ll

def tobias_init_ensemble(ensemble, n_classes, value=100):
    for i in range(1, len(ensemble.models)):
        tobias_initialization(ensemble.models[i], (i-1) % n_classes, value)

def tobias_initialization(model, clss, value=100):
    last_layer = list(module for module in model.modules() if isinstance(module, nn.Linear))[-1]
    last_layer.bias.data[clss] = value

def save_ensemble(ensemble, path):
    for i in range(len(ensemble.models)):
        dict_path = f'{path}_state_dict_{i}.pt'
        torch.save(ensemble.models[i].state_dict(), dict_path)
    rl_path = f'{path}_rls.pt'
    torch.save(ensemble.rls, rl_path)

def load_ensemble(ensemble, path):
    for i in range(len(ensemble.models)):
        dict_path = f'{path}_state_dict_{i}.pt'
        ensemble.models[i].load_state_dict(torch.load(dict_path))
    rl_path = f'{path}_rls.pt'
    ensemble.rls = torch.load(rl_path)

def get_best_gpu():
    u = [torch.cuda.utilization(i) for i in range(torch.cuda.device_count())]
    return f"cuda:{u.index(min(u))}"

@torch.no_grad()
def torch_get_outputs(model, loader, device):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input)), dim=0)
    return outputs, targets

@torch.no_grad()
def torch_get_outputs_destercke(model, loader, device, alpha):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input, alpha, 'euclidean')), dim=0)
    return outputs, targets

def printargs(args):
    print("\n" + "=" * 30)
    print("Starting experiment with: ")
    print("=" * 30)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30 + "\n")
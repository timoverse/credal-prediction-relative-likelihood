import argparse
import json
import pickle

import torch
import numpy as np
from models import get_model, LikelihoodEnsemble, DesterckeEnsemble, CaprioEnsemble, WangEnsemble, Ensemble
import utils
from data import get_data_ood
from probly.quantification.classification import upper_entropy, lower_entropy, upper_entropy_convex_hull, lower_entropy_convex_hull

PATH = '/home/scratch/likelihood-ensembles/'


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

    id_loader, ood_loader = get_data_ood(name_id=args.dataset, name_ood=args.ooddata, seed=args.seed, batch_size=500)

    if args.method == 'ours':
        ensemble = LikelihoodEnsemble(get_model(args.model, args.classes), args.classes, n_members=args.n_members, tobias_value=args.tobias)
        utils.load_ensemble(ensemble, f'{PATH}checkpoints/{args.dataset}_{args.model}_{args.n_members}_{args.tobias}_{args.alpha}_{args.seed}')
    else:
        if args.method == "credalwrapper":
            ensemble = Ensemble(get_model(args.model, args.classes), args.n_members)
        elif args.method == "credalensembling":
            ensemble = DesterckeEnsemble(get_model(args.model, args.classes), args.n_members)
        elif args.method == "credalbnn":
            ensemble = CaprioEnsemble(get_model(args.model, args.classes), args.n_members, args.prior_mu, args.prior_std)
        elif args.method == "credalnet":
            ensemble = WangEnsemble(get_model(args.model, args.classes), args.n_members, args.delta, args.classes)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        ensemble.load_state_dict(torch.load(f'{PATH}new_checkpoints/baseline_{args.dataset}_{args.method}_{args.n_members}_{args.seed}'))

    ensemble = ensemble.to(device)
    ensemble.eval()
    if args.method == 'credalensembling':
        id_outputs, _ = utils.torch_get_outputs_destercke(ensemble, id_loader, device, alpha=args.alpha)
        ood_outputs, _ = utils.torch_get_outputs_destercke(ensemble, ood_loader, device, alpha=args.alpha)
    else:
        id_outputs, _ = utils.torch_get_outputs(ensemble, id_loader, device)
        ood_outputs, _ = utils.torch_get_outputs(ensemble, ood_loader, device)

    id_outputs = id_outputs.detach().cpu().numpy()
    ood_outputs = ood_outputs.detach().cpu().numpy()

    if args.method in ['ours', 'credalwrapper', 'credalnet']:
        id_upper_entropy = upper_entropy(id_outputs)
        ood_upper_entropy = upper_entropy(ood_outputs)
        id_lower_entropy = lower_entropy(id_outputs)
        ood_lower_entropy = lower_entropy(ood_outputs)
    elif args.method in ['credalensembling', 'credalbnn']:
        id_upper_entropy = upper_entropy_convex_hull(id_outputs)
        ood_upper_entropy = upper_entropy_convex_hull(ood_outputs)
        id_lower_entropy = lower_entropy_convex_hull(id_outputs)
        ood_lower_entropy = lower_entropy_convex_hull(ood_outputs)

    ood_dict = {
        'id_upper_entropy': id_upper_entropy,
        'ood_upper_entropy': ood_upper_entropy,
        'id_lower_entropy': id_lower_entropy,
        'ood_lower_entropy': ood_lower_entropy,
    }

    if args.method in ['ours', 'credalensembling']:
        with open(f'{PATH}ood/{args.dataset}_{args.ooddata}_{args.method}_{args.n_members}_{args.alpha}_{args.seed}.pkl', 'wb') as f:
            pickle.dump(ood_dict, f)
    else:
        with open(f'{PATH}ood/{args.dataset}_{args.ooddata}_{args.method}_{args.n_members}_{args.seed}.pkl', 'wb') as f:
            pickle.dump(ood_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='iD Dataset')
    parser.add_argument('--ooddata', type=str, help='OoD Dataset')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--method', type=str, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--n_members', type=int, help='Number of members to use')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--tobias', type=int, help='Tobias <3 value')
    parser.add_argument('--alpha', type=float, help='Alpha value to use')
    parser.add_argument('--delta', type=float, help='Delta value to use')
    parser.add_argument('--prior_mu', type=json.loads, help='Prior mean')
    parser.add_argument('--prior_std', type=json.loads, help='Prior std')
    args = parser.parse_args()
    main(args)

# python ood.py --dataset cifar10 --ooddata svhn --classes 10 --method ours --model resnet --alpha 0.95 --n_members 20 --tobias 100 --seed 1

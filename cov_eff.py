import pickle
from tqdm import tqdm
from models import LikelihoodEnsemble, get_model, Ensemble, DesterckeEnsemble, CaprioEnsemble, WangEnsemble
import torch
import numpy as np
from data import get_data_task
import utils
from probly.metrics import coverage, efficiency, coverage_convex_hull
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

PATH = '/home/scratch/likelihood-ensembles/checkpoints'

def main():
    results_dict = {'ours': {}, 'credalwrapper': {}, 'credalensembling': {}, 'credalbnn': {}, 'credalnet': {}}

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = 'qualitymri'
    base = 'torchresnet'
    seeds = [1, 2, 3]
    tobias = 100
    n_members = 20
    classes = 2
    plt.plot()
    cmap = plt.get_cmap('tab10')
    n = 10
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]

    # OURS
    covs = np.empty((len(seeds), len(alphas)))
    effs = np.empty((len(seeds), len(alphas)))
    accs = np.empty((len(seeds), len(alphas), n_members))
    for j, seed in enumerate(seeds):
        test_loader = get_data_task(dataset, seed=seed)
        for i, alpha in enumerate(alphas):
            ensemble = LikelihoodEnsemble(get_model(base, classes), classes, n_members=n_members, tobias_value=tobias)
            utils.load_ensemble(ensemble, f'{PATH}/{dataset}_{base}_{n_members}_{tobias}_{alpha}_{seed}')
            ensemble = ensemble.to(device)
            ensemble.eval()
            outputs, targets = utils.torch_get_outputs(ensemble, test_loader, device)
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            cov = coverage(outputs, targets)
            eff = efficiency(outputs)

            targets = np.argmax(targets, axis=1)
            for o in range(n_members):
                preds = np.argmax(outputs[:, o, :], axis=1)
                accs[j, i, o] = accuracy_score(targets, preds)
            covs[j, i] = cov
            effs[j, i] = eff
    results_dict['ours']['cov'] = covs.mean(0)
    results_dict['ours']['eff'] = effs.mean(0)
    results_dict['ours']['cov_std'] = covs.std(0)
    results_dict['ours']['eff_std'] = effs.std(0)
    results_dict['ours']['accs'] = accs.mean(0)
    results_dict['ours']['accs_std'] = accs.std(0)

    # CREDALWRAPPER
    covs = np.empty((len(seeds)))
    effs = np.empty((len(seeds)))
    accs = np.empty((len(seeds), n_members))
    for j, seed in enumerate(seeds):
        test_loader = get_data_task(dataset, seed=seed)
        ensemble = Ensemble(get_model(base, classes), n_members)
        ensemble.load_state_dict(torch.load(f'{PATH}/baseline_{dataset}_credalwrapper_{n_members}_{seed}'))
        ensemble = ensemble.to(device)
        ensemble.eval()
        outputs, targets = utils.torch_get_outputs(ensemble, test_loader, device)
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        cov = coverage(outputs, targets)
        eff = efficiency(outputs)
        covs[j] = cov
        effs[j] = eff
        targets = np.argmax(targets, axis=1)
        for o in range(n_members):
            preds = np.argmax(outputs[:, o, :], axis=1)
            accs[j, o] = accuracy_score(targets, preds)
    results_dict['credalwrapper']['cov'] = covs.mean()
    results_dict['credalwrapper']['eff'] = effs.mean()
    results_dict['credalwrapper']['cov_std'] = covs.std()
    results_dict['credalwrapper']['eff_std'] = effs.std()
    results_dict['credalwrapper']['accs'] = accs.mean(0)
    results_dict['credalwrapper']['accs_std'] = accs.std(0)

    # CREDALENSEMBLING
    covs = np.empty((len(seeds), len(alphas) - 1))
    effs = np.empty((len(seeds), len(alphas) - 1))
    accs = np.empty((len(seeds), n_members))
    for j, seed in enumerate(seeds):
        test_loader = get_data_task(dataset, seed=seed)
        ensemble = DesterckeEnsemble(get_model(base, classes), n_members)
        ensemble.load_state_dict(torch.load(f'{PATH}/baseline_{dataset}_credalensembling_{n_members}_{seed}'))
        ensemble = ensemble.to(device)
        ensemble.eval()
        outputs, targets = utils.torch_get_outputs_destercke(ensemble, test_loader, device, 0.0)
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        targets = np.argmax(targets, axis=1)
        for o in range(outputs.shape[1]):
            preds = np.argmax(outputs[:, o, :], axis=1)
            accs[j, o] = accuracy_score(targets, preds)
        for i, alpha in enumerate(alphas[:-1]):
            outputs, targets = utils.torch_get_outputs_destercke(ensemble, test_loader, device, alpha)
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            cov = coverage_convex_hull(outputs, targets)
            eff = efficiency(outputs)
            covs[j, i] = cov
            effs[j, i] = eff
    results_dict['credalensembling']['cov'] = covs.mean(0)
    results_dict['credalensembling']['eff'] = effs.mean(0)
    results_dict['credalensembling']['cov_std'] = covs.std(0)
    results_dict['credalensembling']['eff_std'] = effs.std(0)
    results_dict['credalensembling']['accs'] = accs.mean(0)
    results_dict['credalensembling']['accs_std'] = accs.std(0)

    # CREDALBNN
    covs = np.empty((len(seeds)))
    effs = np.empty((len(seeds)))
    accs = np.empty((len(seeds), n_members))
    for j, seed in enumerate(seeds):
        test_loader = get_data_task(dataset, seed=seed)
        ensemble = CaprioEnsemble(get_model(base, classes), n_members, [0, 1], [0, 1])
        ensemble.load_state_dict(torch.load(f'{PATH}/baseline_{dataset}_credalbnn_{n_members}_{seed}'))
        ensemble = ensemble.to(device)
        ensemble.eval()
        outputs, targets = utils.torch_get_outputs(ensemble, test_loader, device)
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        cov = coverage_convex_hull(outputs, targets)
        eff = efficiency(outputs)
        covs[j] = cov
        effs[j] = eff
        targets = np.argmax(targets, axis=1)
        for o in range(n_members):
            preds = np.argmax(outputs[:, o, :], axis=1)
            accs[j, o] = accuracy_score(targets, preds)

    results_dict['credalbnn']['cov'] = covs.mean()
    results_dict['credalbnn']['eff'] = effs.mean()
    results_dict['credalbnn']['cov_std'] = covs.std()
    results_dict['credalbnn']['eff_std'] = effs.std()
    results_dict['credalbnn']['accs'] = accs.mean(0)
    results_dict['credalbnn']['accs_std'] = accs.std(0)
    print(covs.mean(), effs.mean(), covs.std(), effs.std(), accs.mean(0), accs.std(0))

    # CREDALNET
    covs = np.empty((len(seeds)))
    effs = np.empty((len(seeds)))
    accs = np.empty((len(seeds), n_members))
    base = 'cre' + base
    for j, seed in enumerate(seeds):
        test_loader = get_data_task(dataset, seed=seed)
        ensemble = WangEnsemble(get_model(base, classes), n_members, 0, classes)
        ensemble.load_state_dict(torch.load(f'{PATH}/baseline_{dataset}_credalnet_{n_members}_{seed}'))
        ensemble = ensemble.to(device)
        ensemble.eval()
        outputs = torch.empty(0, device=device)
        targets = torch.empty(0, device=device)
        acc_outputs = torch.empty(0, device=device)
        for input, target in tqdm(test_loader):
            input, target = input.to(device), target.to(device)
            targets = torch.cat((targets, target), dim=0)
            with torch.no_grad():
                outputs = torch.cat((outputs, ensemble.predict_representation(input)), dim=0)
                acc_outputs = torch.cat((acc_outputs, torch.stack([model(input) for model in ensemble.models], dim=1).reshape(len(target), 20, 2, classes)), dim=0)
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        acc_outputs = acc_outputs.detach().cpu().numpy()
        cov = coverage(outputs, targets)
        eff = efficiency(outputs)
        covs[j] = cov
        effs[j] = eff

        targets = np.argmax(targets, axis=1)
        for o in range(n_members):
            preds = np.argmax(utils.intersection_probability(acc_outputs[:, o, :]), axis=1)
            accs[j, o] = accuracy_score(targets, preds)

    results_dict['credalnet']['cov'] = covs.mean()
    results_dict['credalnet']['eff'] = effs.mean()
    results_dict['credalnet']['cov_std'] = covs.std()
    results_dict['credalnet']['eff_std'] = effs.std()
    results_dict['credalnet']['accs'] = accs.mean(0)
    results_dict['credalnet']['accs_std'] = accs.std(0)
    with open(f'{PATH}/{dataset}_result.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == '__main__':
    main()
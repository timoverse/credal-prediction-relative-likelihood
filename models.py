import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from probly.representation import Ensemble, Bayesian
from utils import tobias_init_ensemble
import numpy as np

def get_model(base, n_classes):
    if base == 'resnet':
        model = ResNet18()
        model.linear = nn.Linear(512, n_classes)
    elif base == 'fcnet':
        model = FCNet(768, n_classes)
    elif base == 'torchresnet':
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif base == "creresnet":
        model = ResNet18()
        model.linear = nn.Sequential(
            nn.Linear(model.linear.in_features, 2*n_classes),
            nn.BatchNorm1d(2*n_classes),
            IntSoftmax()
        )
    elif base == "crefcnet":
        model = CreFCNet(768, n_classes)
    elif base == "cretorchresnet":
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2*n_classes),
            nn.BatchNorm1d(2*n_classes),
            IntSoftmax()
        )
    else:
        raise ValueError(f"Unknown base model: {base}")
    return model

class LikelihoodEnsemble(Ensemble):
    def __init__(self, base, n_classes, n_members, tobias_value=100):
        super().__init__(base, n_members)
        self.n_members = n_members
        self.rls = [1.0]
        self.tobias_value = tobias_value
        if self.tobias_value:
            tobias_init_ensemble(self, n_classes, tobias_value)


class CaprioEnsemble(Ensemble):
    def __init__(self, base, n_members, prior_mu, prior_sigma):
        super().__init__(base, n_members)
        self.n_members = n_members
        mus = np.linspace(prior_mu[0], prior_mu[1], endpoint=True, num=n_members)
        sigmas = np.random.uniform(prior_sigma[0], prior_sigma[1], size=n_members)
        for i in range(n_members):
            self.models[i] = Bayesian(base, prior_mean=mus[i], prior_std=sigmas[i])


class DesterckeEnsemble(Ensemble):
    def __init__(self, base, n_members):
        super().__init__(base, n_members)
        self.n_members = n_members

    @torch.no_grad()
    def predict_representation(self, x: torch.Tensor, alpha: float, distance: str, logits: bool = False) -> torch.Tensor:
        x = super().predict_representation(x, logits)
        if distance == 'euclidean':
            # when the distance is euclidean the mean is the representative probability distribution
            representative = torch.mean(x, dim=1)
            # compute distances to the representative distribution
            dists = torch.cdist(x, torch.unsqueeze(representative, 1), p=2)
            # discard alpha percent of the predictions with the largest distances
            # sort the distances
            sorted_indices = torch.argsort(dists.squeeze(), dim=1)
            # get the indices of the predictions to keep
            keep_indices = sorted_indices[:, :int(round((1 - alpha) * self.n_members))]
            # get the predictions to keep
            keep_predictions = torch.gather(x, 1, keep_indices.unsqueeze(2).expand(-1, -1, x.shape[2]))
        else:
            raise ValueError(f"Unknown distance metric: {distance}")
        return keep_predictions


class WangEnsemble(Ensemble):
    def __init__(self, base, n_members, delta, n_classes):
        super().__init__(base, n_members)
        self.n_members = n_members
        self.delta = delta
        self.n_classes = n_classes

    def predict_pointwise(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if logits:
            raise ValueError('Logits not possible for credal nets')
        outputs = torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)
        return outputs.reshape(outputs.shape[0], 2, -1).mean(dim=1)

    def predict_representation(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if logits:
            raise ValueError('Logits not possible for credal nets')
        outputs = torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)
        return outputs.reshape(outputs.shape[0], 2, -1)


class FCNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

class CreFCNet(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2 * n_classes)
        self.bn = nn.BatchNorm1d(2 * n_classes)
        self.act = nn.ReLU()
        self.int_softmax = IntSoftmax()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        x = self.bn(x)
        x = self.int_softmax(x)
        return x

class CreResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(self.base.children())[:-2])
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2 * n_classes)
        self.bn = nn.BatchNorm1d(2 * n_classes)
        self.act = nn.ReLU()
        self.int_softmax = IntSoftmax()

    def forward(self, x):
        x = self.upsample(x)
        x = self.base(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        x = self.int_softmax(x)
        return x

class IntSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Extract number of classes
        n_classes = int(x.shape[-1] / 2)

        # Extract center and the radius
        center = x[:, :n_classes]
        radius = x[:, n_classes:]

        # Ensure the nonnegativity of radius
        radius_nonneg = F.softplus(radius)

        # Compute upper and lower probabilities
        exp_center = torch.exp(center)
        exp_center_sum = torch.sum(exp_center, dim=-1, keepdim=True)

        lo = torch.exp(center - radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center - radius_nonneg))
        hi = torch.exp(center + radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center + radius_nonneg))

        # Generate output
        output = torch.cat([lo, hi], dim=-1)

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
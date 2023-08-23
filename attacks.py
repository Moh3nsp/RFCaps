import torch
from torch.distributions import normal
from utility import COMED_Agg, fedcaps_get_avg
import copy

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bitflipping(model):
    keys = list(model.state_dict().keys())
    for key in keys:
        model.state_dict()[key] = - model.state_dict()[key]
    return model


def apply_byzantine_attack(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu, std = 0, 1
    m = normal.Normal(loc=mu, scale=std)
    for key in model.state_dict().keys():
        shape = model.state_dict()[key].shape
        x = model.state_dict()[key].clone().detach() + torch.tensor(m.sample(sample_shape=shape),
                                                                    device=device).clone().detach()
        model.state_dict()[key].copy_(x)
    return model


def cooperative_attack_for_COMED(models):
    IPMA = copy.deepcopy(models)
    fed_avg_state_dict = fedcaps_get_avg(IPMA)
    for model_index in range(len(models)):
        keys = list(models[model_index].state_dict().keys())
        for key in keys:
            models[model_index].state_dict()[key].copy_(fed_avg_state_dict[key] * 0.16)
    return models




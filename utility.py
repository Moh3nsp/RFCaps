import copy

from torch.distributions import normal
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)

from functools import reduce

def get_avg(models_arr):
    main=models_arr[0][0].state_dict()
    keys=main.keys();

    for index in range(len(models_arr)):
        if(index>0):
            current_model= models_arr[index]
            for key in keys:
                coef=1
                #if(current_model[1] >0):
                #    coef-=current_model[1]
                main[key] += coef* current_model[0].state_dict()[key].clone().detach()# current_model[1]*current_model[0].state_dict()[key]
    for key in keys:
        main[key]  /=    len(models_arr)
    return main
def fedcaps_get_avg(models_arr):
    main=models_arr[0].state_dict()
    keys=main.keys()

    for index in range(len(models_arr)):
        if(index>0):
            current_model= models_arr[index]
            for key in keys:
                main[key] += current_model.state_dict()[key].clone().detach()
    for key in keys:
        main[key]  /=    len(models_arr)
    return main

def fedcaps_get_avg_v3(models_arr):
    main=models_arr[0]
    keys=main.keys()

    for index in range(len(models_arr)):
        if(index>0):
            current_model= models_arr[index]
            for key in keys:
                main[key] += current_model[key].clone().detach()
    for key in keys:
        main[key]  /=    len(models_arr)
    return main

def fed_avg(models_arr):
    main=models_arr[0][0].state_dict()
    keys=main.keys()
    for index in range(len(models_arr)):
        if(index>0):
            current_model= models_arr[index]
            for key in keys:
                main[key] += current_model[0].state_dict()[key]# current_model[1]*current_model[0].state_dict()[key]
    for key in keys:
        main[key]  /=    len(models_arr)
    return main

def fed_avg(models_arr):
    main=models_arr[0][0].state_dict()
    keys=main.keys()
    for index in range(len(models_arr)):
        if(index>0):
            current_model= models_arr[index]
            for key in keys:
                main[key] += current_model[0].state_dict()[key]# current_model[1]*current_model[0].state_dict()[key]
    for key in keys:
        main[key]  /=    len(models_arr)
    return main


def simple_fed_avg(models_arr):
    main=copy.deepcopy( models_arr[0].state_dict())
    for index in range(len(models_arr)):
        if(index>0):
            for layer_name in models_arr[0].state_dict():
                current_model=models_arr[index]
                main[layer_name].copy_( main[layer_name].clone().detach() + current_model.state_dict()[layer_name].clone().detach())

    for layer_name in models_arr[0].state_dict():
        #'primaryCaps.conv.bias'
        main[layer_name].copy_(main[layer_name] / len(models_arr))
    return main


def get_acc_coef(correct_array,real_sample_num_in_each_class , class_value=0.1):
    coef= 0.
    for correct_count_per_class in correct_array:
         coef+= (float(correct_count_per_class)/float(real_sample_num_in_each_class)) * class_value
    return coef



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def select_aggregation(models,selected):
    pass

def get_coordinate_wise_median_of_weights(model_dict, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor

    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})
    #weight_dict== add dimention of model_dict lentgh
    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        median_weight_array = []
        for m in range(len(weight_names_list)):
            current_layer_median = torch.median(weight_dict[weight_names_list[m]], 0).values
            median_weight_array.append(current_layer_median)

    return median_weight_array

def COMED_Agg(main_model, model_dict, device):
    median_weight_array = get_coordinate_wise_median_of_weights(model_dict, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = median_weight_array[j]
    return main_model

def Custom_COMED_Agg(main_model, model_dict, device):
    median_weight_array = get_coordinate_wise_median_of_weights(model_dict, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = median_weight_array[j]
    return main_model

def _flatten(model):
    flatten_weights=[]
    for p in model.parameters():
        flatten_weights.append(torch.flatten(p).detach().cpu().numpy())
    return np.concatenate(tuple(flatten_weights), axis=0)
def index_unique(result,exist):
    final_result=[]
    while(len(final_result)<len(result) ):
        for index in result:
            if( index not in exist):
                final_result.append(int(index))
            else:
                is_same=True
                while(is_same):
                    new_index = int(torch.randint(low=0,high=10,size=(1,1))[0])
                    invalid_arr= list(final_result)+ list(exist)
                    if(new_index not in  invalid_arr):
                        final_result.append(new_index)
                        is_same=False
    return torch.tensor(final_result)
def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min
def scale_list(l, to_min, to_max):
    return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]

def COMED_Killer(model):
    for key in model.state_dict().keys():
        flat=sorted(list(model.state_dict()[key].flatten()))
        start = int(len(flat) / 4)
        end =start*3
        lower= min(flat[start:end])
        higher_bound= max(flat[start:end])
        random_w= torch.randn(model.state_dict()[key].flatten().__len__())
        result=scale_list(random_w, lower, higher_bound)
        model.state_dict()[key] = torch.tensor(result).reshape(model.state_dict()[key].shape)
    return model

def COMED_Killer2(models):
    for key in model.state_dict().keys():
        flat=sorted(list(model.state_dict()[key].flatten()))
        start = int(len(flat) / 4)
        end =start*3
        lower= min(flat[start:end])
        higher_bound= max(flat[start:end])
        random_w= torch.randn(model.state_dict()[key].flatten().__len__())
        result=scale_list(random_w, lower, higher_bound)
        model.state_dict()[key] = torch.tensor(result).reshape(model.state_dict()[key].shape)
    return model

def draw_Acc_Curves(acc_lists):
    #[fedcaps , fed_Avg  , KRUM , COMED]
    proposed_acc=acc_lists[0][0]
    Avg_acc=acc_lists[1][0]
    KRUM_acc=acc_lists[2][0]
    COMED_acc=acc_lists[3][0]

    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

    plt.title('Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.plot([i for i in range(len(proposed_acc))], proposed_acc, '-d', markerfacecolor="None",
             label='proposed algorithm', c='g')

    plt.plot([i for i in range(len(Avg_acc))], Avg_acc, '-o', markerfacecolor="None",
             label='Fed-Avg', c='r')

    plt.plot([i for i in range(len(KRUM_acc))], KRUM_acc, '-d', markerfacecolor="None",
             label='KRUM', c='b')

    plt.plot([i for i in range(len(COMED_acc))], COMED_acc, '-o', markerfacecolor="None",
             label='COMED', c='k')

    plt.legend('name of attack : ',acc_lists[0][1] ,'number of attackers : ' ,acc_lists[0][2])
    plt.show()














def trimmed_mean(w, trim_ratio):
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


def tmean(b,inputs):
    if len(inputs) - 2 * b > 0:
        b = b
    else:
        b = b
        while len(inputs) - 2 * b <= 0:
            b -= 1
        if b < 0:
            raise RuntimeError

    stacked = torch.stack(inputs, dim=0)
    largest, _ = torch.topk(stacked, b, 0)
    neg_smallest, _ = torch.topk(-stacked, b, 0)
    new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
    new_stacked /= len(inputs) - 2 * b
    return new_stacked
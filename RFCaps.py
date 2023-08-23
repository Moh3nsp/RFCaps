import torch
from torch.utils.data import DataLoader
from torch import tensor
import copy
import numpy as np

from fedcaps import GlobalFedCaps, LocalFedCapsNet, fed_train, test
from database import noise_get_attacker_train_dataset, get_train_dataset, get_attacker_train_dataset, get_test_dataset, get_testset_dataloader
from utility import COMED_Agg, trimmed_mean, tmean, _flatten, fed_avg, index_unique, COMED_Killer, draw_Acc_Curves
from attacks import bitflipping, apply_byzantine_attack, cooperative_attack_for_COMED
from tools.krum import Krum as krum_aggregation
from fecCaps_agg import fed_Caps_agg

################# Fixed #######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
non_iid_percentage = 80
routing_iterations = 3
class_count = 10
avg_loss_margin = 0.1

global_model_test = get_testset_dataloader(800)
global_test_set_dataloader = DataLoader(
    global_model_test, batch_size=500, shuffle=True)
sample_num = 10
attackers_per_communication = []

##################################################

participant_number = tensor(10)

# models initialize
# conv1_Channel,
# primary_channel,
# capsule_dimention,
# kernel_size,
# primaryCaps_filter_size

algorithms_dict = {4: 'TMean', 0: 'FedCaps',
                   1: 'Fed_Avg', 2: 'KRUM', 3: 'COMED'}

attacks_dict = {0: 'byzantine', 1: 'label_flipping', 2: 'data_noise'}

number_of_attacks_dict = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    5: 23
}

algg_global_acc = {name: [] for index, name in algorithms_dict.items()}

for alg_index, alg_name in algorithms_dict.items():
    for attack_index, attack_name in attacks_dict.items():
        for num_attack_index, att_num in number_of_attacks_dict.items():

            global_model = LocalFedCapsNet(32, 2, 8, 11, 11).to(device)

            global_model2 = LocalFedCapsNet(256, 32, 8, 9, 9).to(device)

            init_weights = global_model.state_dict()

            local_models = [[LocalFedCapsNet(32, 2, 8, 11, 11).to(device), False]
                            for i in range(participant_number)]

            # init weights
            for index in range(len(local_models)):
                local_models[index][0].load_state_dict(init_weights)

            # dataset load
            datasets = []
            random_lables = torch.randint(
                low=0, high=class_count, size=(1, participant_number))[0]

            for lbl_name in random_lables:
                datasets.append(
                    (get_train_dataset(lbl_name, sample_num, non_iid_percentage), False))

            datanoise_attacks_count = 0
            datanoise_attacks_Index = []

            if attack_index == 2:
                datanoise_attacks_count = att_num
                datanoise_attacks_Index = torch.randperm(participant_number)[
                    :datanoise_attacks_count]

                for i in datanoise_attacks_Index:
                    local_models[i][1] = True

                lbl_index = 0
                for i in datanoise_attacks_Index:
                    datasets[int(i)] = (noise_get_attacker_train_dataset(
                        lbl_index, sample_num, non_iid_percentage), True)
                lbl_index += 1

            ########################### Byzantine Attack ####################
            byzantine_attacks_count = 0
            byzantine_attacks_Index = []
            if (attack_index == 0):
                byzantine_attacks_count = att_num
                byzantine_attacks_Index = torch.randperm(participant_number)[
                    :byzantine_attacks_count]
                byzantine_attacks_Index = index_unique(
                    byzantine_attacks_Index, datanoise_attacks_Index)

                for i in byzantine_attacks_Index:
                    local_models[i][1] = True

            ##################### label flipping attack #####################

            lbl_name_source = [
                [(0, 8), (3, 8), (8, 0)],
                [(0, 9), (0, 3), (9, 0)],
                [(2, 5), (6, 8), (6, 4)],
                [(3, 8), (0, 8), (0, 9)],
                [(1, 8), (1, 8), (1, 9)]
            ]

            lbl_name_from_to = [lbl_name_source[int(torch.randint(
                low=0, high=len(lbl_name_source), size=(1, 1))[0])] for i in range(att_num)]
            lbl_flipping_attacker_count = 0
            lbl_flipping_attacker_Index = []

            if attack_index == 1:
                lbl_flipping_attacker_count = att_num
                lbl_flipping_attacker_Index = torch.randperm(
                    participant_number)[:lbl_flipping_attacker_count]
                lbl_flipping_attacker_Index = index_unique(
                    lbl_flipping_attacker_Index, byzantine_attacks_Index)

                for i in lbl_flipping_attacker_Index:
                    local_models[i][1] = True

                lbl_index = 0
                for i in lbl_flipping_attacker_Index:
                    datasets[int(i)] = (get_attacker_train_dataset(
                        lbl_name_from_to[lbl_index], sample_num, non_iid_percentage), True)
                lbl_index += 1

            # load training test set

            total_attacker_counts = lbl_flipping_attacker_count + \
                byzantine_attacks_count + datanoise_attacks_count

            total_attacker_index = list(lbl_flipping_attacker_Index) + list(
                byzantine_attacks_Index) + list(datanoise_attacks_Index)

            models_datasets = {index: [item[0], item[1]]
                               for index, item in enumerate(zip(local_models, datasets))}

            clustered_models = {i: [] for i in range(class_count)}

            total_comunications = 100

            is_first_comunicate = True

            local_training_count = 1
            is_first_comunicate_Fedcaps_Agg = True

            global_total_acc = []

            for comunication_iter_index in range(total_comunications):
                # Train
                if (is_first_comunicate == False):
                    local_training_count = 1

                for _ in range(local_training_count):
                    print(alg_name, '_', attack_name, '_', att_num, '_',
                          'local training round :', comunication_iter_index)
                    for model_index in range(len(models_datasets)):
                        data_loader = DataLoader(
                            models_datasets[model_index][1][0], batch_size=20, shuffle=True)
                        models_datasets[model_index][0][0] = fed_train(models_datasets[model_index][0][0],
                                                                       data_loader
                                                                       )
                if attack_index == 0:
                    models = [models_datasets[int(byzindex)][0][0]
                              for byzindex in byzantine_attacks_Index]
                    attacker_modeles = cooperative_attack_for_COMED(models)
                    loop_index = 0
                    for byz_index in byzantine_attacks_Index:
                        models_datasets[int(
                            byz_index)][0][0] = attacker_modeles[loop_index]
                        loop_index += 1

                number_of_attackers = 0

                if (alg_name == 'KRUM'):
                    # KRUM aggregation
                    weights = [torch.tensor(_flatten(
                        model[0][0]), dtype=torch.float) for _, model in models_datasets.items()]
                    weights = [w[torch.randperm(len(w))[:10000]]
                               for w in weights]
                    krum_Agg = krum_aggregation(
                        n=participant_number, f=total_attacker_counts, m=participant_number - total_attacker_counts)
                    agg_indices = krum_Agg(inputs=weights)
                    models_for_agg = [
                        data[0] for key, data in models_datasets.items() if key in agg_indices]
                    avg_weights = fed_avg(models_for_agg)
                    global_model.load_state_dict(avg_weights)

                    for index in agg_indices:
                        if index in total_attacker_index:
                            number_of_attackers += 1

                    attackers_per_communication.append(number_of_attackers)

                if (alg_name == "COMED"):
                    model_dict = {
                        _: model_dataset[0][0] for _, model_dataset in models_datasets.items()}
                    global_model = COMED_Agg(global_model, model_dict, device)

                if (alg_name == "TMean"):
                    model_dict = {
                        _: model_dataset[0][0] for _, model_dataset in models_datasets.items()}
                    inputs = [model_dataset[0][0].state_dict()
                              for _, model_dataset in models_datasets.items()]
                    global_model_weight = trimmed_mean(inputs, 0.1)
                    global_model.load_state_dict(global_model_weight)

                if (alg_name == "Fed_Avg"):
                    models = [model_dataset[0]
                              for _, model_dataset in models_datasets.items()]
                    global_model.load_state_dict(fed_avg(models))
                    number_of_attackers = len(total_attacker_index)

                if (alg_name == "FedCaps"):
                    model_dict = [model_dataset[0][0]
                                  for _, model_dataset in models_datasets.items()]
                    avg_weights, clustered_models = fed_Caps_agg(
                        model_dict, clustered_models, is_first_comunicate_Fedcaps_Agg, participant_number, "DIGITMNIST", 20)
                    global_model.load_state_dict(avg_weights)
                    is_first_comunicate_Fedcaps_Agg = False

                for index in range(len(models_datasets)):
                    models_datasets[index][0][0].load_state_dict(
                        global_model.state_dict())

                final_loss_val = test(
                    "global model", global_model, global_test_set_dataloader)

                print('final accuracy of global model in iteration  : ',
                      comunication_iter_index,
                      ' is  : ',
                      final_loss_val[0].item(),
                      'Accuracy: ',
                      final_loss_val[2].item())

                global_total_acc.append(final_loss_val[2].item())

            torch.save(
                global_total_acc, f"{alg_name}_{attack_name}_{participant_number}_{att_num}_global_acc.pth")

            algg_global_acc[alg_name].append(
                (global_total_acc, attack_name, att_num))


#[fedcaps , fed_Avg  , KRUM , COMED]

#algorithms_dict= {0:'FedCaps', 1:'Fed_Avg' , 2:'KRUM' , 3:'COMED'}
index_plt = 0
for attack_index, attack_name in attacks_dict.items():
    for num_attack_index, att_num in number_of_attacks_dict.items():

        plt_data = [
            algg_global_acc['FedCaps'][index_plt],
            algg_global_acc['Fed_Avg'][index_plt],
            algg_global_acc['KRUM'][index_plt],
            algg_global_acc['COMED'][index_plt]
        ]
        draw_Acc_Curves(plt_data)
        index_plt += 1


# for att_name in ['byzantine','label_flipping','data_noise']:
#  for att_count in [5,10,15,20,23]:
#
#    comed =torch.load(f'COMED_{att_name}_50_{att_count}_global_acc.pth')
#    krum=torch.load(f'KRUM_{att_name}_50_{att_count}_global_acc.pth')
#    FedCAps = torch.load(f'FedCaps_{att_name}_50_{att_count}_global_acc.pth')
#
#    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
#
#    plt.plot([i for i in range(len(krum))], krum, '--', markerfacecolor="None",
#              label='KRUM', c='b')
#    plt.plot([i for i in range(len(comed))], comed, '--', markerfacecolor="None",
#              label='COMED', c='k')
#    plt.plot([i for i in range(len(FedCAps))], FedCAps , '--', markerfacecolor="None",
#              label='FedCAps', c='r')
#
#    plt.title(f'Learning Curve with {att_count} {att_name} attackers')
#
#    plt.xlabel('Iteration')
#
#    plt.ylabel('Accuracy')
#
#
#    plt.legend()
#    plt.show()

import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
from utility import AddGaussianNoise

torch.manual_seed(0)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],0
    #                     std=[0.229, 0.224, 0.225] )
        ])

def get_testset_dataloader(per_class_sample_count):
    finaltest_set = torch.utils.data.dataset.ConcatDataset([get_test_dataset(0, count=per_class_sample_count),
                                                            get_test_dataset(1, count=per_class_sample_count),
                                                            get_test_dataset(2, count=per_class_sample_count),
                                                            get_test_dataset(3, count=per_class_sample_count),
                                                            get_test_dataset(4, count=per_class_sample_count),
                                                            get_test_dataset(5, count=per_class_sample_count),
                                                            get_test_dataset(6, count=per_class_sample_count),
                                                            get_test_dataset(7, count=per_class_sample_count),
                                                            get_test_dataset(8, count=per_class_sample_count),
                                                            get_test_dataset(9, count=per_class_sample_count)
                                                            ])

    return finaltest_set



def FashionMNIST_get_testset_dataloader(per_class_sample_count):
    finaltest_set = torch.utils.data.dataset.ConcatDataset([FashionMNIST_get_test_dataset(0, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(1, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(2, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(3, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(4, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(5, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(6, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(7, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(8, count=per_class_sample_count),
                                                            FashionMNIST_get_test_dataset(9, count=per_class_sample_count)
                                                            ])

    return finaltest_set


def CIFAR10_get_testset_dataloader(per_class_sample_count):
    finaltest_set = torch.utils.data.dataset.ConcatDataset([cifar10_get_test_dataset(0, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(1, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(2, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(3, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(4, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(5, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(6, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(7, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(8, count=per_class_sample_count),
                                                            cifar10_get_test_dataset(9, count=per_class_sample_count)
                                                            ])

    return finaltest_set


def CIFAR10_total_testset_dataloader():
    transform = torchvision.transforms.ToTensor()
    dataset_test = torchvision.datasets.CIFAR10(root='./datasets', download=True, train=False, transform=transform)
    return dataset_test




def noise_get_attacker_train_dataset(lblId, count , non_iid_data__percentage):
    #attack_labelIds_tuple [(from_lbl , to_lbl)]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0.,2)
    ])

    non_iid_count = int((count * non_iid_data__percentage) / 100)
    iid_count = int(count - non_iid_count)
    dataset_train = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transforms)
    iid_dataset_train = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transforms)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        dataset_train.targets = dataset_train.targets[indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train

    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)
    idx = idx_index[0][torch.randperm(non_iid_count)]
    dataset_train.targets = dataset_train.targets[idx]
    dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]

    if (non_iid_data__percentage == 0 or non_iid_data__percentage == 100):
        return dataset_train

    return torch.utils.data.ConcatDataset([dataset_train, iid_dataset_train])


def FashionMNIST_noise_get_attacker_train_dataset(lblId, count , non_iid_data__percentage):
    #attack_labelIds_tuple [(from_lbl , to_lbl)]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0.,2)
    ])

    non_iid_count = int((count * non_iid_data__percentage) / 100)
    iid_count = int(count - non_iid_count)
    dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True, transform=transforms)
    iid_dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True, transform=transforms)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        dataset_train.targets = dataset_train.targets[indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train

    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)
    idx = idx_index[0][torch.randperm(non_iid_count)]
    dataset_train.targets = dataset_train.targets[idx]
    dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]

    if (non_iid_data__percentage == 0 or non_iid_data__percentage == 100):
        return dataset_train

    return torch.utils.data.ConcatDataset([dataset_train, iid_dataset_train])



def FashionMNIST_get_attacker_train_dataset(attack_labelIds_tuple, count , non_iid_data__percentage):

    #attack_labelIds_tuple [(from_lbl , to_lbl)]

    non_iid_count =  (count * non_iid_data__percentage) //100
    non_iid_count_per_each_attack =    non_iid_count // len(attack_labelIds_tuple)
    iid_count=int( count-non_iid_count)
    iid_dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True, transform=transform)
    dataset_list=[]


    iid_dataset_train.targets=torch.tensor( iid_dataset_train.targets)

    for atk_index in range(len(attack_labelIds_tuple)):
        dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True,transform=transform)
        dataset_train.targets = torch.tensor(dataset_train.targets)

        idx = dataset_train.targets == attack_labelIds_tuple[atk_index][0]
        idx_index = torch.where(idx == True)
        idx = idx_index[0][torch.randperm(non_iid_count_per_each_attack)]
        dataset_train.targets =  [attack_labelIds_tuple[atk_index][1] for i in range(non_iid_count_per_each_attack)]
        dataset_train.data = dataset_train.data[idx]
        dataset_list.append(dataset_train)


    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]
    if(non_iid_data__percentage != 0 or non_iid_data__percentage !=100):
        dataset_list.append(iid_dataset_train)
    return torch.utils.data.ConcatDataset( dataset_list)

def FashionMNIST_get_train_dataset(lblId, count,non_iid_data__percentage,is_attack=False,fake_label=0,poison_percentage=0):
    non_iid_count =int( (count * non_iid_data__percentage) /100 )

    iid_count=int( count-non_iid_count)

    dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True, transform=transform)
    iid_dataset_train = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=True, transform=transform)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        dataset_train.targets = dataset_train.targets[indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train
    dataset_train.targets=torch.tensor( dataset_train.targets)
    iid_dataset_train.targets = torch.tensor(iid_dataset_train.targets)
    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)
    idx = idx_index[0][torch.randperm(non_iid_count)]
    if(is_attack==False):
        dataset_train.targets = dataset_train.targets[idx]
        dataset_train.data = dataset_train.data[idx]
    else:
         must_fake_count=int((len(idx) *poison_percentage) / 100);
         must_real_count =-int((len(idx) -must_fake_count))
         new_lbl = [fake_label.item() for i in range(must_fake_count)]
         for item in  dataset_train.targets[idx[must_real_count: ]]:
             new_lbl.append(item.item())
         dataset_train.targets =new_lbl
         dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]
    if(non_iid_data__percentage==0 or non_iid_data__percentage==100):
        return dataset_train
    return torch.utils.data.ConcatDataset([ dataset_train,iid_dataset_train])

def FashionMNIST_get_test_dataset(lblId=-1,count=10):
    dataset_test = torchvision.datasets.FashionMNIST(root='./datasets', download=True, train=False, transform=transform)

    dataset_test.targets=torch.tensor( dataset_test.targets)
    if(lblId==-1) :
        indices = torch.randperm(len(dataset_test))[:count]
        dataset_test.targets = dataset_test.targets[indices]
        dataset_test.data = dataset_test.data[indices]
        return dataset_test

    idx = dataset_test.targets ==lblId
    idx_index= torch.where(idx==True)
    idx= idx_index[0][ torch.randperm( count)]
    dataset_test.targets = dataset_test.targets[idx]
    dataset_test.data = dataset_test.data[idx]
    return  dataset_test

def cifar10_get_attacker_train_dataset(attack_labelIds_tuple, count , non_iid_data__percentage):
    #attack_labelIds_tuple [(from_lbl , to_lbl)]
    transform = torchvision.transforms.ToTensor()
    per_attack_count=count/len(attack_labelIds_tuple)
    dataset_list=[]

    for atk_index in range(len(attack_labelIds_tuple)):
        dataset_train = torchvision.datasets.CIFAR10(root='./datasets', download=True, train=True,transform=transform)
        dataset_train.targets = torch.tensor(dataset_train.targets)
        idx = dataset_train.targets == attack_labelIds_tuple[atk_index][0]
        idx_index = torch.where(idx == True)
        idx = idx_index[0][:per_attack_count]
        dataset_train.targets =  [attack_labelIds_tuple[atk_index][1] for i in range(per_attack_count)]
        dataset_train.data = dataset_train.data[idx]
        dataset_list.append(dataset_train)

    return torch.utils.data.ConcatDataset( dataset_list)

def cifar10_get_train_dataset(lblId, count,non_iid_data__percentage,is_attack=False,fake_label=0,poison_percentage=0):
    non_iid_count =int( (count * non_iid_data__percentage) /100 )
    iid_count=int( count-non_iid_count)

    transform = torchvision.transforms.ToTensor()
    dataset_train = torchvision.datasets.CIFAR10(root='./datasets', download=True, train=True, transform=transform)
    iid_dataset_train = torchvision.datasets.CIFAR10(root='./datasets', download=True, train=True, transform=transform)
    dataset_train.targets = torch.tensor(dataset_train.targets)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        dataset_train.targets = dataset_train.targets[ indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train
    dataset_train.targets=torch.tensor( dataset_train.targets)
    iid_dataset_train.targets = torch.tensor(iid_dataset_train.targets)
    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)
    idx = idx_index[0][torch.randperm(non_iid_count)]
    if(is_attack==False):
        dataset_train.targets = dataset_train.targets[idx]
        dataset_train.data = dataset_train.data[idx]
    else:
         must_fake_count=int((len(idx) *poison_percentage) / 100);
         must_real_count =-int((len(idx) -must_fake_count))
         new_lbl = [fake_label.item() for i in range(must_fake_count)]
         for item in  dataset_train.targets[idx[must_real_count: ]]:
             new_lbl.append(item.item())
         dataset_train.targets =new_lbl
         dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]
    if(non_iid_data__percentage==0 or non_iid_data__percentage==100):
        return dataset_train
    return torch.utils.data.ConcatDataset([ dataset_train,iid_dataset_train])

def cifar10_get_test_dataset(lblId=-1,count=10):
    transform = torchvision.transforms.ToTensor()
    dataset_test = torchvision.datasets.CIFAR10(root='./datasets', download=True, train=False, transform=transform)

    dataset_test.targets=torch.tensor( dataset_test.targets)
    if(lblId==-1) :
        indices = torch.randperm(len(dataset_test))[:count]
        dataset_test.targets = dataset_test.targets[indices]
        dataset_test.data = dataset_test.data[indices]
        return dataset_test

    idx = dataset_test.targets ==lblId
    idx_index= torch.where(idx==True)
    idx= idx_index[0][ torch.randperm( count)]
    dataset_test.targets = dataset_test.targets[idx]
    dataset_test.data = dataset_test.data[idx]
    return  dataset_test


def EMNIST_noise_get_attacker_train_dataset(lblId, count , non_iid_data__percentage):
    #attack_labelIds_tuple [(from_lbl , to_lbl)]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0.,2)
    ])

    non_iid_count = int((count * non_iid_data__percentage) / 100)
    iid_count = int(count - non_iid_count)
    dataset_train = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True, transform=transforms)
    iid_dataset_train = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True, transform=transforms)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        #indices =torch.tensor( [x for x in indices if x not in test_ix])
        dataset_train.targets = dataset_train.targets[indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train

    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)[0]
    #idx_index = torch.tensor( [x for x in idx_index if x not in test_ix])
    idx = idx_index[torch.randperm(non_iid_count)]
    dataset_train.targets = dataset_train.targets[idx]
    dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]

    if (non_iid_data__percentage == 0 or non_iid_data__percentage == 100):
        return dataset_train

    return torch.utils.data.ConcatDataset([dataset_train, iid_dataset_train])

def EMNIST_get_attacker_train_dataset(attack_labelIds_tuple, count , non_iid_data__percentage):
    #attack_labelIds_tuple [(from_lbl , to_lbl)]
    transform = torchvision.transforms.ToTensor()
    per_attack_count=int(count/len(attack_labelIds_tuple))
    dataset_list=[]

    for atk_index in range(len(attack_labelIds_tuple)):
        dataset_train = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True,transform=transform)
        dataset_train.targets = torch.tensor(dataset_train.targets)
        idx = dataset_train.targets == attack_labelIds_tuple[atk_index][0]
        idx_index = torch.where(idx == True)[0]
        #idx_index =torch.tensor( [x for x in idx_index if x not in test_ix])
        idx = idx_index[:int(per_attack_count)]
        dataset_train.targets =  [attack_labelIds_tuple[atk_index][1] for i in range(per_attack_count)]
        dataset_train.data = dataset_train.data[idx]
        dataset_list.append(dataset_train)

    return torch.utils.data.ConcatDataset( dataset_list)

def EMNIST_get_train_dataset(test_ix,lblId, count,non_iid_data__percentage,is_attack=False,fake_label=0,poison_percentage=0):
    non_iid_count =int( (count * non_iid_data__percentage) /100 )
    iid_count=int( count-non_iid_count)

    transform = torchvision.transforms.ToTensor()
    dataset_train = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True, transform=transform)
    iid_dataset_train = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True, transform=transform)
    dataset_train.targets = torch.tensor(dataset_train.targets)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        indices= torch.tensor( [x for x in indices if x not in  test_ix])
        dataset_train.targets = dataset_train.targets[ indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train

    dataset_train.targets=torch.tensor( dataset_train.targets)
    iid_dataset_train.targets = torch.tensor(iid_dataset_train.targets)
    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True )[0]
    idx_index =torch.tensor( [x for x in idx_index if x not in  test_ix])


    idx = idx_index[torch.randperm(non_iid_count)]
    dataset_train.targets = dataset_train.targets[idx]
    dataset_train.data = dataset_train.data[idx]


    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]

    if(non_iid_data__percentage==0 or non_iid_data__percentage==100):
        return dataset_train
    return torch.utils.data.ConcatDataset([ dataset_train,iid_dataset_train]) # , list(idx)+list(iid_indices)

def EMNIST_get_test_dataset(count=260):
    transform = torchvision.transforms.ToTensor()
    dataset_test = torchvision.datasets.EMNIST(root='./datasets',split='letters', download=True, train=True, transform=transform)

    dataset_test.targets=torch.tensor( dataset_test.targets)
    indices = torch.randperm(len(dataset_test))[:count]
    dataset_test.targets = dataset_test.targets[indices]
    dataset_test.data = dataset_test.data[indices]
    return dataset_test,indices

def EMNIST_get_testset_dataloader(test_ix,per_class_sample_count):
    dataset=[EMNIST_get_train_dataset(test_ix,lbl, count=per_class_sample_count) for lbl in range(26)]
    finaltest_set = torch.utils.data.dataset.ConcatDataset([])
    return finaltest_set

def get_attacker_train_dataset(attack_labelIds_tuple, count , non_iid_data__percentage):
    # attack_labelIds_tuple [(from_lbl , to_lbl)]
    transform = torchvision.transforms.ToTensor()
    per_attack_count = int(count / len(attack_labelIds_tuple))
    dataset_list=[]
    for atk_index in range(len(attack_labelIds_tuple)):
        dataset_train = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
        dataset_train.targets = torch.tensor(dataset_train.targets)
        idx = dataset_train.targets == attack_labelIds_tuple[atk_index][0]
        idx_index = torch.where(idx == True)
        idx = idx_index[0][:per_attack_count]
        dataset_train.targets = [attack_labelIds_tuple[atk_index][1] for i in range(per_attack_count)]
        dataset_train.data = dataset_train.data[idx]
        dataset_list.append(dataset_train)

    return torch.utils.data.ConcatDataset(dataset_list)

def get_train_dataset(lblId, count,non_iid_data__percentage,is_attack=False,fake_label=0,poison_percentage=0):
    non_iid_count =int( (count * non_iid_data__percentage) /100 )
    iid_count=int( count-non_iid_count)
    dataset_train = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
    iid_dataset_train = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
    if (lblId == -1):
        indices = torch.randperm(len(dataset_train))[:count]
        dataset_train.targets = dataset_train.targets[indices]
        dataset_train.data = dataset_train.data[indices]
        return dataset_train

    idx = dataset_train.targets == lblId
    idx_index = torch.where(idx == True)
    idx = idx_index[0][torch.randperm(non_iid_count)]
    if(is_attack==False):
        dataset_train.targets = dataset_train.targets[idx]
        dataset_train.data = dataset_train.data[idx]
    else:
         must_fake_count=int((len(idx) *poison_percentage) / 100);
         must_real_count =-int((len(idx) -must_fake_count))
         new_lbl = [fake_label.item() for i in range(must_fake_count)]
         for item in  dataset_train.targets[idx[must_real_count: ]]:
             new_lbl.append(item.item())
         dataset_train.targets =new_lbl
         dataset_train.data = dataset_train.data[idx]

    iid_indices = torch.randperm(len(iid_dataset_train))[:iid_count]
    iid_dataset_train.targets = iid_dataset_train.targets[iid_indices]
    iid_dataset_train.data = iid_dataset_train.data[iid_indices]
    if(non_iid_data__percentage==100):
        return dataset_train
    return torch.utils.data.ConcatDataset([ dataset_train,iid_dataset_train])

def get_test_dataset(lblId=-1,count=10):
    dataset_test = torchvision.datasets.MNIST(root='./datasets', download=True, train=False, transform=transform)

    if(lblId==-1) :
        indices = torch.randperm(len(dataset_test))[:count]
        dataset_test.targets = dataset_test.targets[indices]
        dataset_test.data = dataset_test.data[indices]
        return dataset_test

    idx = dataset_test.targets ==lblId
    idx_index= torch.where(idx==True)
    idx= idx_index[0][ torch.randperm( count)]
    dataset_test.targets = dataset_test.targets[idx]
    dataset_test.data = dataset_test.data[idx]
    return  dataset_test

def get_dataloader(dataset,batch_size):
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)

def get_mnist_loader(label_num,batch_size=10):
    train_data=  get_dataloader(get_train_dataset(label_num),batch_size)
    test_data=   get_dataloader(get_test_dataset(label_num), batch_size)
    return(train_data,test_data)

def get_mnist_dataset(lbl_num ,count):
    return get_train_dataset(lbl_num,count)





class MyDataset(Dataset):
    def __init__(self,data, transform=None):

        self.data1= [d1[0] for d1 in data]
        self.data2 = [d2[1] for d2 in data]
        self.targets= [t[2] for t in data]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((100, 256)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize([0.485 ], [0.229 ])
                      ])

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = self.targets[index]

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1,x2, y

    def __len__(self):
        return len(self.data1)
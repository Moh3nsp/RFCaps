from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import sys
import argparse
import functools
sys.argv = ['']
del sys

caps_number = 0

# conv1_Channel=32
# primary_channel=2 # timeto 8
# capsule_dimention=8


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)
        s = None
        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)
                c = F.softmax(b_batch.view(-1, output_caps)
                              ).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)
        return v, c


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(
            input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(
            0), self.input_caps, self.output_caps, self.output_dim)
        v, c = self.routing_module(u_predict)
        u_predict = None
        return v, c


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps *
                              output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class LocalFedCapsNet(nn.Module):
    def __init__(self, conv1_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size, routing_iterations=3, n_classes=10):
        super(LocalFedCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            1, conv1_Channel, kernel_size=kernel_size, stride=1)

        self.primaryCaps = PrimaryCapsLayer(
            conv1_Channel, primary_channel, capsule_dimention, kernel_size=primaryCaps_filter_size, stride=2)

        self.caps_number = self.primaryCaps(
            F.relu(self.conv1(torch.randint(0, 1, (1, 1, 28, 28), dtype=torch.float32))))
        self.caps_number = self.caps_number.shape[1]
        # primary_channel * number_primaryCaps * number_primaryCaps
        self.num_primaryCaps = self.caps_number
        self.routing_module = AgreementRouting(
            self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(
            self.num_primaryCaps, capsule_dimention, n_classes, 16, self.routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x, c = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class CIFAR10_LocalFedCapsNet(nn.Module):
    def __init__(self, conv1_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size, routing_iterations=3, n_classes=10):
        super(CIFAR10_LocalFedCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, conv1_Channel, kernel_size=kernel_size, stride=1)

        self.primaryCaps = PrimaryCapsLayer(
            conv1_Channel, primary_channel, capsule_dimention, kernel_size=primaryCaps_filter_size, stride=2)

        self.caps_number = self.primaryCaps(
            F.relu(self.conv1(torch.randint(0, 1, (1, 3, 32, 32), dtype=torch.float32))))
        self.caps_number = self.caps_number.shape[1]
        # primary_channel * number_primaryCaps * number_primaryCaps
        self.num_primaryCaps = self.caps_number
        self.routing_module = AgreementRouting(
            self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(
            self.num_primaryCaps, capsule_dimention, n_classes, 16, self.routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x, c = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class GlobalFedCaps(nn.Module):
    def __init__(self,
                 conv1_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size,
                 routing_iterations=3,
                 n_classes=10, ensimble_num=0):
        super(GlobalFedCaps, self).__init__()

        # added with one for avg

        self.ensimble_count = ensimble_num

        self.conv_1 = nn.ModuleList([nn.Conv2d(
            1, conv1_Channel, kernel_size=kernel_size, stride=1) for i in range(ensimble_num)])

        self.primaryCaps = nn.ModuleList([PrimaryCapsLayer(
            conv1_Channel, primary_channel, capsule_dimention, kernel_size=primaryCaps_filter_size, stride=2) for i in range(ensimble_num)])
        local_model = LocalFedCapsNet(
            conv1_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size)
        # ensimble_num * primary_channel * number_primaryCaps * number_primaryCaps
        self.num_primaryCaps = local_model.caps_number*ensimble_num

        self.routing_module = AgreementRouting(
            self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(
            self.num_primaryCaps, capsule_dimention, n_classes, 16, self.routing_module)

    def forward(self, input):
        conv_outs = [F.relu(self.conv_1[cnv_index](input))
                     for cnv_index in range(self.ensimble_count)]
        primarycaps_outs = [self.primaryCaps[cnv_index](conv_outs[cnv_index])
                            for cnv_index in range(self.ensimble_count)]
        capsules = torch.cat((primarycaps_outs), dim=1)

        x, c = self.digitCaps(capsules)

        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class Cifar10_GlobalFedCaps(nn.Module):
    def __init__(self, conv1_Channel, conv2_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size,
                 routing_iterations=3,
                 n_classes=10, ensimble_num=0):
        super(Cifar10_GlobalFedCaps, self).__init__()

        # added with one for avg

        self.ensimble_count = ensimble_num

        self.conv_1 = nn.ModuleList([nn.Conv2d(
            1, conv1_Channel, kernel_size=kernel_size, stride=1) for i in range(ensimble_num)])
        self.conv_2 = nn.ModuleList([nn.Conv2d(
            conv1_Channel, conv2_Channel, kernel_size=kernel_size, stride=1) for i in range(ensimble_num)])

        fc_input_size = self.conv_2[0](self.conv_1[0](
            torch.randint(0, 1, (1, 1, 32, 32), dtype=torch.float32)))
        in_size = int(fc_input_size.flatten().shape[0]*3)
        self.fc1 = nn.Linear(in_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, 10)

        self.primaryCaps = nn.ModuleList([PrimaryCapsLayer(
            conv2_Channel, primary_channel, capsule_dimention, kernel_size=primaryCaps_filter_size, stride=2) for i in range(ensimble_num)])
        local_model = CIFAR10_LocalFedCapsNet(
            conv1_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size)
        # ensimble_num * primary_channel * number_primaryCaps * number_primaryCaps
        self.num_primaryCaps = local_model.caps_number*ensimble_num

        self.routing_module = AgreementRouting(
            self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(
            self.num_primaryCaps, capsule_dimention, n_classes, 16, self.routing_module)

    def forward(self, input, is_fc):
        if (is_fc):
            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)
            self.fc3.requires_grad_(True)
            self.fc_out.requires_grad_(True)

            self.conv_1[:].requires_grad_(False)
            self.conv_2[:].requires_grad_(False)
            self.primaryCaps[:].requires_grad_(False)
            self.routing_module.requires_grad_(False)
            self.digitCaps.requires_grad_(False)

            data_list = [input[:, 0, :, :][:, None, :, :], input[:, 1, :, :][:, None, :, :],
                         input[:, 2, :, :][:, None, :, :]]
            conv_outs = [F.relu(self.conv_1[cnv_index](data_list[cnv_index]))
                         for cnv_index in range(self.ensimble_count)]
            conv_outs2 = [F.relu(self.conv_2[cnv_index](conv_outs[cnv_index]))
                          for cnv_index in range(self.ensimble_count)]

            fc_in_size = []
            for o in conv_outs2:
                fc_in_size += list(o.flatten())
            out = self.fc1(torch.tensor(fc_in_size))
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc_out(out)
            return out

        else:
            return self.forward_main(self, input)

    def forward_main(self, input):
        data_list = [input[:, 0, :, :][:, None, :, :], input[:, 1,
                                                             :, :][:, None, :, :], input[:, 2, :, :][:, None, :, :]]
        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc3.weight.requires_grad = False
        self.fc_out.weight.requires_grad = False

        self.conv_1[:].requires_grad_(True)
        self.conv_2[:].requires_grad_(True)
        self.primaryCaps[:].requires_grad_(True)
        self.routing_module.requires_grad_(True)
        self.digitCaps.requires_grad_(True)

        conv_outs = [F.relu(self.conv_1[cnv_index](data_list[cnv_index]))
                     for cnv_index in range(self.ensimble_count)]
        conv_outs2 = [F.relu(self.conv_2[cnv_index](conv_outs[cnv_index]))
                      for cnv_index in range(self.ensimble_count)]
        primarycaps_outs = [self.primaryCaps[cnv_index](conv_outs2[cnv_index])
                            for cnv_index in range(self.ensimble_count)]
        capsules = torch.cat((primarycaps_outs), dim=1)

        x, c = self.digitCaps(capsules)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class Cifar10_GlobalFedCapsv2(nn.Module):
    def __init__(self,
                 conv1_Channel, conv2_Channel, primary_channel, capsule_dimention, kernel_size, primaryCaps_filter_size,
                 routing_iterations=3,
                 n_classes=10, ensimble_num=0):
        super(Cifar10_GlobalFedCapsv2, self).__init__()

        # added with one for avg

        self.ensimble_count = ensimble_num

        self.conv_1 = nn.Conv2d(
            3, conv1_Channel, kernel_size=kernel_size, stride=1)
        self.conv_2 = nn.Conv2d(
            conv1_Channel, conv2_Channel, kernel_size=kernel_size, stride=1)

        fc_input_size = self.conv_2(self.conv_1(
            torch.randint(0, 1, (1, 3, 32, 32), dtype=torch.float32)))
        in_size = int(fc_input_size.flatten().shape[0])
        self.fc1 = nn.Linear(in_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, 10)

        self.primaryCaps = PrimaryCapsLayer(
            conv2_Channel, primary_channel, capsule_dimention, kernel_size=primaryCaps_filter_size, stride=2)
        local_model = self.primaryCaps(self.conv_2(self.conv_1(
            torch.randint(0, 1, (1, 3, 32, 32), dtype=torch.float32))))
        # ensimble_num * primary_channel * number_primaryCaps * number_primaryCaps
        self.num_primaryCaps = local_model.shape[1]

        self.routing_module = AgreementRouting(
            self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(
            self.num_primaryCaps, capsule_dimention, n_classes, 16, self.routing_module)

    def forward(self, input, is_fc):
        if (is_fc):
            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)
            self.fc3.requires_grad_(True)
            self.fc_out.requires_grad_(True)

            self.conv_1.requires_grad_(False)
            self.conv_2.requires_grad_(False)
            self.primaryCaps.requires_grad_(False)
            self.routing_module.requires_grad_(False)
            self.digitCaps.requires_grad_(False)

            conv_outs = F.relu(self.conv_1(input))
            conv_outs2 = F.relu(self.conv_2(conv_outs))
            out_cnn = torch.flatten(conv_outs2, 1)
            out = self.fc1(out_cnn)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc_out(out)
            return out

        else:
            return self.forward_main(input)

    def forward_main(self, input):

        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc3.weight.requires_grad = False
        self.fc_out.weight.requires_grad = False

        self.conv_1.requires_grad_(True)
        self.conv_2.requires_grad_(True)
        self.primaryCaps.requires_grad_(True)
        self.routing_module.requires_grad_(True)
        self.digitCaps.requires_grad_(True)

        conv_outs = F.relu(self.conv_1(input))
        conv_outs2 = F.relu(self.conv_2(conv_outs))

        capsules = self.primaryCaps(conv_outs2)
        x, c = self.digitCaps(capsules)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros(
            (x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        # losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
      #           self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)

        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2

        margin_loss = targets * left + self.lambda_ * (1. - targets) * right
        margin_loss = margin_loss.mean()

        return margin_loss

      #  return losses.mean() if size_average else losses.sum()


parser = argparse.ArgumentParser(description='CapsNet with MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--routing_iterations', type=int, default=3)
parser.add_argument('--with_reconstruction',
                    action='store_true', default=False)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


def margin_loss2(x, targets, size_average=True):
    batch_size = x.size(0)
    t = torch.zeros(x.size()).long()
    if targets.is_cuda:
        t = t.cuda()
    v_c = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))

    left = F.relu(0.9 - v_c).view(batch_size, -1)
    right = F.relu(v_c - 0.1).view(batch_size, -1)
    t = t.scatter_(1, targets.data.view(-1, 1), 1)
    targets = Variable(t)
    loss = targets * left + 0.5 * (1.0 - targets) * right
    loss = loss.sum(dim=1).mean()

    return loss
    # if __name__ == '__main__':


def test(model_name, model, test_loader, reconstruction_alpha=0.0001, number_of_Class=10):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    correct_pred_count = [0 for i in range(number_of_Class)]
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(
            torch.tensor(target, requires_grad=False))

        if args.with_reconstruction:
            output, probs = model(data)
            #reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
            reconstruction_loss = F.mse_loss(output, data.view(-1, 1024))
            test_loss += loss_fn(probs, target, size_average=False).data
            test_loss += reconstruction_alpha * reconstruction_loss
        else:
            # props is a list of probabilities
            # (20,10 ) [[0.2,0.6],.....]
            # each array of 20 contains prob of target in this array
            # you can take argmax and get predict of this data

            output, probs = model(data)
            test_loss += loss_fn(probs, target, size_average=False).data
            del output
        # pred is a list of final prediction each element is a number of predicted label
        # get the index of the max probability
        pred = probs.data.max(1, keepdim=True)[1]
        # pred_Eq is a list of True False prediction
        pred_Eq = pred.eq(target.data.view_as(pred))
        correct += pred_Eq.cpu().sum()
        for p_index in range(len(target)):
            if (pred_Eq[p_index]):
                correct_pred_count[target[p_index]] += 1

    acc = 100*(correct/len(test_loader.dataset))
    # acc_of_current_cluster
    # acc_of_other

    return test_loss, correct_pred_count, acc


def evaluation(model_name, model, test_loader, reconstruction_alpha=0.0001, number_of_Class=10):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    correct_pred_count = [0 for i in range(number_of_Class)]
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(
            torch.tensor(target, requires_grad=False))

        if args.with_reconstruction:
            output, probs = model(data)
            #reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
            reconstruction_loss = F.mse_loss(output, data.view(-1, 1024))
            test_loss += loss_fn(probs, target, size_average=False).data
            test_loss += reconstruction_alpha * reconstruction_loss
        else:
            # props is a list of probabilities
            # (20,10 ) [[0.2,0.6],.....]
            # each array of 20 contains prob of target in this array
            # you can take argmax and get predict of this data

            output, probs = model(data)
            test_loss += loss_fn(probs, target, size_average=False).data
            del output
        # pred is a list of final prediction each element is a number of predicted label
        # get the index of the max probability
        pred = probs.data.max(1, keepdim=True)[1]
        # pred_Eq is a list of True False prediction
        pred_Eq = pred.eq(target.data.view_as(pred))
        correct += pred_Eq.cpu().sum()
        for p_index in range(len(target)):
            if (pred_Eq[p_index]):
                correct_pred_count[target[p_index]] += 1

    acc = 100*(correct/len(test_loader.dataset))
    # acc_of_current_cluster
    # acc_of_other

    return test_loss, correct_pred_count, acc


def fed_train(model: GlobalFedCaps, train_loader, reconstruction_alpha=0.0001):
    model.train()
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        if args.with_reconstruction:
            output, probs = model(data)
            #reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
            reconstruction_loss = F.mse_loss(output, data.view(-1, 1024))
            margin_loss = loss_fn(probs, target)
            loss = reconstruction_alpha * reconstruction_loss + margin_loss
        else:
            output, probs = model(data)
            loss = loss_fn(probs, target)
        loss.backward()
        optimizer.step()
    return (model, loss)


def cifar10_test(model_name, model: Cifar10_GlobalFedCaps, test_loader, reconstruction_alpha=0.0001, number_of_Class=10):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    correct_pred_count = [0 for i in range(number_of_Class)]
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(
            torch.tensor(target, requires_grad=False))

        if args.with_reconstruction:
            output, probs = model(data, False)
            #reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
            reconstruction_loss = F.mse_loss(output, data.view(-1, 1024))
            test_loss += loss_fn(probs, target, size_average=False).data
            test_loss += reconstruction_alpha * reconstruction_loss
        else:
            # props is a list of probabilities
            # (20,10 ) [[0.2,0.6],.....]
            # each array of 20 contains prob of target in this array
            # you can take argmax and get predict of this data

            output, probs = model(data, False)
            test_loss += loss_fn(probs, target, size_average=False).data
            del output
        # pred is a list of final prediction each element is a number of predicted label
        # get the index of the max probability
        pred = probs.data.max(1, keepdim=True)[1]
        # pred_Eq is a list of True False prediction
        pred_Eq = pred.eq(target.data.view_as(pred))
        correct += pred_Eq.cpu().sum()
        for p_index in range(len(target)):
            if (pred_Eq[p_index]):
                correct_pred_count[target[p_index]] += 1

    acc = 100*(correct/len(test_loader.dataset))
    # acc_of_current_cluster
    # acc_of_other

    return test_loss, correct_pred_count, acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cifar10_fed_train(model: Cifar10_GlobalFedCaps, train_loader, reconstruction_alpha=0.0001):
    model.train()
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss1 = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        pred = model(data, is_fc=True)
        label = torch.tensor([list(torch.eye(10)[int(tg)])
                             for tg in target], device=device)
        loss1_ = loss1(label, pred)
        loss1_.backward()
        optimizer.step()
        optimizer.zero_grad()
        output, probs = model(data, is_fc=False)
        loss = loss_fn(probs, target)
        loss.backward()
        optimizer.step()
    return model


def voting_test(models, test_loader, reconstruction_alpha=0.0001):
    # model.eval()
    test_loss_model = {index: 0 for index in range(len(models))}
    correct = 0
    loss_fn = MarginLoss(0.9, 0.1, 0.5)
    correct_pred_count = [0 for i in range(10)]
    model_data_pred = {index: [] for index in range(len(models))}
    voting_pred = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(
            torch.tensor(target, requires_grad=False))

        if args.with_reconstruction:
            output, probs = model(data)
            reconstruction_loss = F.mse_loss(
                output, data.view(-1, 784), size_average=False).data
            test_loss += loss_fn(probs, target, size_average=False).data
            test_loss += reconstruction_alpha * reconstruction_loss
        else:
            props_per_model = {index: 0 for index in range(len(models))}
            model_index = 0
            for model in models:
                output, props_per_model[model_index] = model(data)
                test_loss_model[model_index] += loss_fn(props_per_model[model_index], target,
                                                        size_average=False).data
                model_index += 1

            # pred is a list of final prediction each element is a number of predicted label
        for mode_index_ in range(len(models)):
            pred = props_per_model[int(mode_index_)].data.max(1, keepdim=True)[
                1]  # get the index of the max probability
            model_data_pred[mode_index_] += pred

        # calc predict
        per_model_pred = {data_index: [] for data_index in range(len(target))}
        for data_index in range(len(target)):
            for model_pred in range(len(model_data_pred)):
                per_model_pred[data_index] += model_data_pred[model_pred][data_index]

        final_pred = []
        for data_index in range(len(target)):
            final_pred.append(torch.mode(torch.tensor(
                per_model_pred[data_index]), 0).values)

        final_pred = torch.tensor(final_pred)
        pred_Eq = final_pred.eq(target.data.view_as(final_pred))

        correct += pred_Eq.cpu().sum()
        for p_index in range(len(target)):
            if (pred_Eq[p_index]):
                correct_pred_count[target[p_index]] += 1

    acc = 100*(correct/len(test_loader.dataset))
    # acc_of_current_cluster
    # acc_of_other

    return correct_pred_count, acc

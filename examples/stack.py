"Stack Data Structure trained from a reference implementation"
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import itertools

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PushNet(nn.Module):
    def __init__(self, name, stack_channels=1, img_channels=3):
        super(PushNet, self).__init__()
        in_channels = stack_channels + img_channels
        out_channels = stack_channels
        nf = 16
        self.name = name
        self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.convmid = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, stack_channels, 3, padding=1)

    def forward(self, *x):
        if len(x) > 1:
            x = torch.cat(x, dim=1)
        # import pdb; pdb.set_trace()
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return (x,)


class PopNet(nn.Module):
    def __init__(self, name, stack_channels=1, img_channels=3):
        super(PopNet, self).__init__()
        self.stack_channels = stack_channels
        self.img_channels = img_channels
        out_channels = stack_channels + img_channels
        self.name = name
        nf = 16
        self.conv1 = nn.Conv2d(stack_channels, nf, 3, padding=1)
        self.convmid = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, out_channels, 3, padding=1)

    def forward(self, *x):
        channel_dim = 1
        x, = x
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        (img, stack) = x.split(self.img_channels, channel_dim)
        return (stack, img)


def net_stack(img_channels):
    """Construct a stack abstract data type"""
    push = PushNet("push", img_channels=img_channels)
    pop = PopNet("pop", img_channels=img_channels)
    empty = Variable(torch.rand(1, 1, 28, 28), requires_grad=True)
    constants = {'empty': empty}
    interface = {'push': push, 'pop': pop}
    # FIXME!
    return {'constants': constants, 'interface': interface}


def list_push(stack, element):
    stack = stack.copy()
    stack.append(element)
    return (stack, )


def list_pop(stack):
    stack = stack.copy()
    item = stack.pop()
    return (stack, item)


def py_stack():
    empty_stack = []
    return {'constants': {'empty': empty_stack},
            'interface': {'push': list_push, 'pop': list_pop}}


def stack_trace(model, items, expand_empty, batch_size):
    """Example stack trace"""
    items = [Variable(data[0]) for data in list(itertools.islice(items, 3))]
    print(len(items))
    push = model['interface']['push']
    pop = model['interface']['pop']
    empty = model['constants']['empty']
    if expand_empty:
        empty = empty.expand(batch_size, 1, 28, 28)
    observes = []
    stack = empty
    # print("mean", items[0].mean())
    (stack,) = push(stack, items[0])
    (stack,) = push(stack, items[1])
    (stack,) = push(stack, items[2])
    (pop_stack, pop_item) = pop(stack)
    observes.append(pop_item)
    (pop_stack, pop_item) = pop(pop_stack)
    observes.append(pop_item)
    return observes


def model_params(model):
    iparams = []
    for (name, net) in model['interface'].items():
        for params in net.parameters():
            iparams.append(params)
    cparams = [var for (name, var) in model['constants'].items()]
    return iparams + cparams


def train(trainloader, reference, model, batch_size):
    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    import pdb; pdb.set_trace()
    optimizer = optim.Adam(model_params(model), lr=0.001)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        dataiter = iter(trainloader)
        refiter = iter(trainloader)
        while True:
            try:
                # get the inputs
                observes = stack_trace(model, dataiter, True, batch_size)
                refobserves = stack_trace(reference, refiter, False, batch_size)

                total_loss = 0.0
                for i in range(len(observes)):
                    loss = criterion(observes[i], refobserves[i])
                    total_loss += loss
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                loss = total_loss
                loss.backward()
                optimizer.step()
                print(total_loss)

                # print statistics
                running_loss += loss.data[0]
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            except StopIteration:
                print("End of epoch")
                continue

    print('Finished Training')


def main(argv):
    batch_size = 32
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=1)
    img_channels = 1
    train(trainloader, py_stack(), net_stack(img_channels), batch_size)


if __name__ == "__main__":
    main(sys.argv[1:])

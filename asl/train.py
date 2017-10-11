import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from tensorboardX import SummaryWriter

def observe_loss(criterion, obs, refobs):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  for (i, _) in enumerate(obs):
    loss = criterion(obs[i], refobs[i])
    total_loss += loss
  return total_loss


def print_stats(i, epoch, running_loss, total_loss):
  "Print optimization statistics"
  print(total_loss)
  if i % 2000 == 1999:    # print every 2000 mini-batches
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2000))
    running_loss = 0.0

def model_params(model):
  iparams = []
  for (name, net) in model.items():
    for params in net.parameters():
      iparams.append(params)
  return iparams


def train(trace, trainloader, reference, model, batch_size,
          nepochs=10, callbacks=[]):
  "Train model using reference wrt to trace"
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model_params(model), lr=0.0001)
  writer = SummaryWriter()

  i = 0
  for epoch in range(nepochs):
    running_loss = 0.0
    dataiter = iter(trainloader)
    refiter = iter(trainloader)
    while True:
      try:
        # get the inputs
        observes = trace(dataiter, **model)
        refobserves = trace(refiter, **reference)
        loss = observe_loss(criterion, observes, refobserves)

        # Optimization Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print_stats(i, epoch, running_loss, loss)
        for callback in callbacks:
          callback(i=i, model=model, reference=reference, writer=writer,
                   loss=loss)
        running_loss += loss.data[0]
        i += 1
      except (StopIteration, IndexError):
        break
  writer.close()
  print('Finished Training')

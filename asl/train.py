import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from asl.callbacks import print_stats, every_n

logstate = {}


def log(name, value):
  logstate[name] = value

def log_append(name, value):
  if name in logstate:
    logstate[name].append(value)
  else:
    logstate[name] = [value]


def reset_log():
  global logstate
  logstate.clear()


def getlog():
  return logstate

def observe_loss(criterion, obs, refobs, state=None):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  losses = [criterion(obs[i], refobs[i]) for i in range(len(obs))]
  print([loss[0].data[0] for loss in losses])
  total_loss = sum(losses)
  return total_loss


def model_params(model):
  iparams = []
  for (name, net) in model.items():
    for params in net.parameters():
      iparams.append(params)
  return iparams


def train(trace,
          trainloader,
          reference,
          model,
          batch_size,
          callbacks=None,
          nepochs=10,
          resetlog=True):
  "Train model using reference wrt to trace"
  print(callbacks)
  callbacks = callbacks if [] is None else callbacks
  callbacks = callbacks + [every_n(print_stats, 100)]
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

        for callback in callbacks:
          callback(i=i,
                   model=model,
                   reference=reference,
                   writer=writer,
                   loss=loss,
                   epoch=epoch,
                   running_loss=running_loss,
                   log=getlog())
        running_loss += loss.data[0]
        i += 1
        if resetlog:
          reset_log()
      except (StopIteration, IndexError):
        break
  writer.close()
  print('Finished Training')

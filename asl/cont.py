import math

def converged(every, print_change=True, change_thres=-0.000005):
  "Has the optimization converged?"
  def converged_gen(every):
    running_loss = 0.0
    last_running_loss = 0.0
    show_change = False
    cont = True
    while True:
      data = yield cont
      if data.loss is None:
        continue
      running_loss += data.loss
      if (data.i + 1) % every == 0:
        if show_change:
          change = (running_loss - last_running_loss)
          print('absolute change (avg over {}) {}'.format(every, change))
          if last_running_loss != 0:
            relchange = change / last_running_loss
            per_iter = relchange / every
            print('relative_change: {}, per iteration: {}'.format(relchange,
                                                                  per_iter))
            if per_iter > change_thres:
              print("Relative change insufficeint, stopping!")
              cont = False
        else:
          show_change = True
        last_running_loss = running_loss
        running_loss = 0.0

  gen = converged_gen(every)
  next(gen)
  return gen


def convergedperc(every, print_change=True, change_thres=0.00001):
  "Converged when threshold is less than percentage? Use when optimum is zero"
  def converged_gen(every):
    running_loss = 0.0
    last_running_loss = 0.0
    show_change = False
    cont = True
    while True:
      data = yield cont
      if data.loss is None:
        continue
      running_loss += data.loss
      if (data.i + 1) % every == 0:
        if show_change:
          if running_loss == 0:
            print("Loss is zero, stopping!")
          else:
            abchange = running_loss - last_running_loss
            percchange = running_loss / last_running_loss
            print('absolute change (avg over {}) {}'.format(every, abchange))
            print('Percentage change (avg over {}) {}'.format(every, percchange))
            per_iter = (1 - percchange) / every
            print('percentage_: {}, per iteration: {}'.format(percchange, per_iter))
            print("What?", per_iter, change_thres, per_iter < change_thres)
            if per_iter < change_thres:
              print("Percentage change insufficient (< {})".format(change_thres))
              cont = False
        else:
          show_change = True
        last_running_loss = running_loss
        running_loss = 0.0

  gen = converged_gen(every)
  next(gen)
  return gen

def convergedmin(every, print_change=True, change_thres=0.000005):
  "Converged when threshold is less than percentage? Use when optimum is zero"
  def converged_gen(every):
    running_loss = math.inf
    last_running_loss = math.inf
    show_change = False
    cont = True
    while True:
      data = yield cont
      if data.loss is None:
        continue
      running_loss = min(data.loss, running_loss)
      if (data.i + 1) % every == 0:
        if show_change:
          if running_loss == 0:
            print("Loss is zero, stopping!")
          else:
            abchange = running_loss - last_running_loss
            percchange = running_loss / last_running_loss
            print('absolute change of min (avg over {}) {}'.format(every, abchange))
            print('Percentage change of min (avg over {}) {}'.format(every, percchange))
            per_iter = (1 - percchange) / every
            print('percentage_: {}, per iteration: {}'.format(percchange, per_iter))
            print("What?", per_iter, change_thres, per_iter < change_thres)
            if per_iter < change_thres:
              print("Percentage change insufficient (< {})".format(change_thres))
              cont = False
        else:
          show_change = True
        last_running_loss = running_loss
        # running_loss = 0.0

  gen = converged_gen(every)
  next(gen)
  return gen


def nancancel(loss, **kwargs):
  "Stop if loss is Nan"
  if (loss != loss):
    print("Loss is NAN: ", loss, " stopping!")
    return True
  else:
    return False

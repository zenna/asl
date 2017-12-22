"Get reference loss"
import asl

def isidle(runstate):
  return runstate['mode'] == "idle"


def empty_runstate():
  return {'observes' : {},
          'mode' : 'idle'}


def set_mode(runstate, mode):
  runstate['mode'] = mode


def mode(runstate):
  return runstate['mode']


def set_idle(runstate):
  set_mode(runstate, 'idle')


def observe(value, label, runstate, log=True):
  if isidle(runstate):
    print("cant observe values without choosing mode")
    raise ValueError

  if runstate['mode'] not in runstate['observes']:
    runstate['observes'][runstate['mode']] = {}

  runstate['observes'][runstate['mode']][label] = value
  return value


def callfuncs(functions, inputs, modes):
  """Execute each function and record runstate"""
  runstate = empty_runstate()
  for i, func in enumerate(functions):
    set_mode(runstate, modes[i])
    func(*inputs[i], runstate)
    set_idle(runstate)
  return runstate

def refresh_iter(dl, itr_transform=None):
  if itr_transform is None:
    return iter(dl)
  else:
    return asl.util.misc.imap(itr_transform, iter(dl))

def run_observe(functions, inputs, refresh_inputs, modes, log=True):
  """Run functions and accumulate observed values
  Args:
    functions
    inputs
  Returns:
    A function of no arguments that produces a ``runstate``, which accumulates
    information from running all the functions in ``functions``
  """
  inp = [refresh_inputs(inp) for inp in inputs]

  def runobserve():
    nonlocal inp
    try:
      runstate = callfuncs(functions, inp, modes)
      if log:
        asl.log("runstate", runstate)
      return runstate
    except StopIteration:
      print("End of Epoch, restarting inputs")
      inp = [refresh_inputs(inp) for inp in inputs]
      return callfuncs(functions, inp, modes)

  return runobserve

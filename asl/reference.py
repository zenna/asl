"Get reference loss"
from enum import Enum
import asl
from asl.loss import vec_dist, dist

class Mode(Enum):
  NOMODE = 0  # No observations allowed
  REF = 1     # Executing function in reference mode
  NEURAL = 2  # Neural mode

GLOBAL_MODE = Mode.NOMODE

def mode():
  return GLOBAL_MODE

def set_mode(md):
  global GLOBAL_MODE
  GLOBAL_MODE = md


REF_OBSERVES = {}
OBSERVES = {}

def get_observes():
  global OBSERVES
  return OBSERVES

def get_ref_observes():
  global REF_OBSERVES
  return REF_OBSERVES

def clear_observes():
  global REF_OBSERVES
  global OBSERVES
  REF_OBSERVES = {}
  OBSERVES = {}

def reference_loss(observes, ref_observes):
  "Loss between observed and references"
  nobs = len(ref_observes)
  for k in observes.keys():
    if k not in ref_observes.keys():
      print("No observes found for ", k)
      raise ValueError
  if nobs == 0:
    raise ValueError

  obvals = [observes[k] for k in observes.keys()]
  refvals = [ref_observes[k] for k in observes.keys()]
  return vec_dist(obvals, refvals)


def observe(value, label):
  global REF_OBSERVES
  global OBSERVES
  if mode() is Mode.NOMODE:
    print("cant observe values without choosing mode")
    raise ValueError
  elif mode() is Mode.NEURAL:
    OBSERVES[label] = value
  else:
    assert mode() is Mode.REF
    REF_OBSERVES[label] = value

  return value


def runref(model, ref_model, items_iter, ref_items_iter):
  "Run f1 and f2 and extract equalities"
  if mode() is not Mode.NOMODE:
    raise ValueError
  clear_observes()

  # First run reference implementation
  set_mode(Mode.NEURAL)
  model(items_iter)

  # Now run reference implementation
  set_mode(Mode.REF)
  ref_model(ref_items_iter)

  # Set to default mode
  set_mode(Mode.NOMODE)
  return OBSERVES, REF_OBSERVES

def inner_loss_gen(model, reference, items_iter, ref_items_iter):
  obs, ref_obs = runref(model, reference, items_iter, ref_items_iter)
  loss = reference_loss(obs, ref_obs)
  return loss


def ref_loss_gen(model, reference, dl, itr_transform=None, loss=dist):
  "Function minimizes difference between reference and neural implementation"
  if itr_transform is None:
    itr = iter
  else:
    itr = lambda loader: asl.util.misc.imap(itr_transform, iter(loader))
  items_iter = itr(dl)
  ref_items_iter = itr(dl)

  def loss_gen():
    nonlocal items_iter, ref_items_iter
    try:
      loss = inner_loss_gen(model, reference, items_iter, ref_items_iter)
      return loss
    except StopIteration:
      print("End of Epoch, restarting iterators")
      clear_observes()
      items_iter = itr(dl)
      ref_items_iter = itr(dl)
      set_mode(Mode.NOMODE)
      loss = inner_loss_gen(model, reference, items_iter, ref_items_iter)
      return loss

  return loss_gen

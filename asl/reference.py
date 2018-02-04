"Get reference loss"
import asl
from asl.loss import dist, mean


def ref_losses(runstate):
  "Map from label to loss"
  observes = runstate['observes']['model']
  ref_observes = runstate['observes']['reference']

  nobs = len(ref_observes)
  for k in observes.keys():
    if k not in ref_observes.keys():
      print("No observes found for ", k)
      raise ValueError
  if nobs == 0:
    raise ValueError

  return {k: dist(observes[k], ref_observes[k]) for k in observes.keys()}

def ref_loss_gen(model, reference, input, refresh_input):
  "Function minimizes difference between reference and neural implementation"
  runobserve = asl.run_observe([model, reference],
                               [input, input],
                               refresh_input,
                               ['model', 'reference'])
  def lossgen():
    runstate = runobserve()
    return ref_losses(runstate)
  return lossgen

def single_ref_loss(model, reference, input, refresh_input, accum=mean):
  "Function minimizes difference between reference and neural implementation"
  runobserve = asl.run_observe([model, reference],
                               [input, input],
                               refresh_input,
                               ['model', 'reference'])
  def lossgen():
    runstate = runobserve()
    losses = ref_losses(runstate)
    return accum(losses.values())
  return lossgen

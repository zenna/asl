import z3
import torch
from multipledispatch import dispatch
MAX_DIM = 5

SymTensor = z3.DeclareSort('SymTensor')
SymShape = z3.DeclareSort('SymShape')
Shape = z3.Function('Shape', SymTensor, z3.IntSort(), SymShape)
ten = Const
z3.Forall

x = Const('x', SymTensor)
if i <= ndims:
  shapedims(i) == 0
axioms = [ForAll([x], Shape(x, Ndims(x)Implies(Human(x), Mortal(x))), 
Human(socrates)

@

def assert_args(args, sym_params, solver):
  for arg, value in args.items():
    sym_param == sym_params[arg]
    solver.add(sym_param == value)

def syn_compose(modulegens, moduleargs):
  solver = z3.Solver()
  ins = [Const("i_0_{}".format(j), SymTensor) for j in range(modulegens[0].n_ins())]
  for modulegen in modulegens:
    outs = [Const("i_{}_{}".format(i+1, j), SymTensor) for j in range(modulegen.n_outs())] 
    modulegen.sym_forward(ins, outs, solver=solver)
    ins = outs


def test_composition():
  in_sizes = [(1, 10, 10), (1, 5, 5)]
  out_sizes = [(10,)]
  rand_in = [torch.rand(sz) for sz in in_sizes]


  module = compose([archs.Normalize,
                    archs.CombineNet,
                    nn.Reshape,
                    nn.Linear,
                    F.softmax],
                    [{"in_sizes": in_sizes},
                     {"out_channels": 5},
                     {"out_sizes": out_sizes},
                     {"out_sizes": out_sizes}],
                     rand_in)
  in_data = [torch.rand(sz) for sz in in_sizes]
  output = module(*in_data)

if __name__ == "__main__":
  test_composition()
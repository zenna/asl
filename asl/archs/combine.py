import asl
from torch import nn

class CombineNet(nn.Module):
  def __init__(self,
               *,
               in_sizes,
               out_sizes):
    "Combines multiple inputs into one input"
    super(CombineNet, self).__init__()
    normalized_size = max(in_sizes, key=asl.archs.nelem)
    normalizers = [asl.archs.normalizer(size, normalized_size)(size, normalized_size)for size in in_sizes]
    self.size_normalizers = nn.ModuleList(normalizers)
  
  def forward(self, *xs):
    xs_same_size = [self.size_normalizers[i](x) for i, x in enumerate(xs)]
    x = self.combine_inputs(xs_same_size)

  def sym_forward(self, args):
    if "in_sizes" in args:
      return {"out_sizes": in_sizes}


class TestLinear(nn.Module):
  def __init__(self,
               *,
               nin,
               nout):
    "Combines multiple inputs into one input"
    super(TestLinear, self).__init__()
    self.layer = nn.Linear(nin, nout)

  def forward(self, x):
    self.layer(x)

  def sym_forward(self, sym_ins, sym_outs, solver):
    sym_in = sym_ins[0]
    sym_out = sym_outs[0]

    self.in_features = Int("in_features")
    self.out_features = Int("out_features")
    self.bias = Bool("bias")
    # shape(y) = out_features
    # nparams = in_features * out_features
    self.sym_params =  {"in_features": in_features,
                        "out_features": out_features,
                        "bias": bias}

    solver.add(self.in_features > 0)
    solver.add(self.out_features > 0)
    solver.add(Ndims(sym_in) > 1)
    solver.add(Shape(sym_out, 0) == Shape(sym_in, 0))
    for i = 1:MAX_DIM:
      z3.If(Ndim(sym_in) == i,
            Shape(sym_out, 0) == Shape(sym_in, 0)))
      Shape(sym_out, i)
    solver.add(Shape(sym_out)[0] = Shape() = out_features


class TestReshape(nn.Module):
  def __init__(self,
               in_ten,
               shape):
    "Combines multiple inputs into one input"
    super(TestReshape, self).__init__()

  def forward(self, x):
    self.view(x)

  def assert_args(self, args, solver):
    for arg, value in args.items()
      sym_param == self.sym_params[arg]
      solver.add(sym_param == arg_value)

  def sym_forward(self, sym_ins, sym_outs, solver):
    (sym_in_ten, sym_in_shape) = sym_in
    sym_out = sym_outs[0]
    solver.add(Nelem(sym_in) == Nelem(sym_out))
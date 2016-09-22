from __future__ import print_function

from dddt.templates import *
from dddt.distances import *
import time
import sys
import numpy as np
from io import *
from dddt.io import placeholder
from dddt.config import floatX
from dddt.distances import *

# import theano
# import theano.tensor as T
# import lasagne
# from lasagne.utils import floatX
# from theano common.variable theano import function
# from theano import confi  g

sys.setrecursionlimit(40000)


def typed_arg_name(type_name, arg_name):
    return "%s::%s" % (arg_name, type_name)


class Type():
    def __init__(self, shape, name, dtype=floatX):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def tensor(self, name='', add_batch=False):
        tensor_name = typed_arg_name(self.name, name)
        # Create a tensor for this shape
        ndims = len(self.shape)
        if add_batch:
            ndims += 1
        broadcastable = (False,)*ndims
        tensor = placeholder(dtype=self.dtype,
                             shape=self.get_shape(add_batch=True),
                             ndim=ndims, name=name)
        return tensor

    def tensor_tf(self, name='', add_batch=False):
        tensor_name = typed_arg_name(self.name, name)
        return tf.placeholder(tf.float32, shape=self.shape, name=tensor_name)

    def get_shape(self, add_batch=False, batch_size=None):
        if add_batch:
            return (batch_size,) + self.shape
        else:
            return self.shape

class Interface():
    def __init__(self, lhs, rhs, name, **template_kwargs):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.template = template = template_kwargs['template']
        self.template_kwargs = template_kwargs
        self.inputs = [type.tensor(add_batch=True, name=self.input_name(type, i))
                       for i, type in enumerate(lhs)]
        # print(self.inputs[0].ndim)
        params = Params()
        self.inp_shapes = [type.get_shape(add_batch=True) for type in lhs]
        self.out_shapes = [type.get_shape(add_batch=True) for type in rhs]
        # output_args = {'batch_norm_update_averages' : True,
        #                'batch_norm_use_averages' : True}
        output_args = {'deterministic': True}
        outputs, params = template(self.inputs, out_shapes=self.out_shapes,
                                   output_args=output_args,
                                   params=params, inp_shapes=self.inp_shapes,
                                   **self.template_kwargs)
        params.lock()
        self.params = params
        self.outputs = outputs

    def __call__(self, *raw_args):
        args = [arg.input_var if hasattr(arg, 'input_var') else arg for arg in raw_args]
        print("Calling", args)
        # output_args = {'batch_norm_update_averages' : True, 'batch_norm_use_averages' : False}
        output_args = {'deterministic': True}
        outputs, params = self.template(*args, output_args=output_args,
                                        params=self.params,
                                        inp_shapes=self.inp_shapes,
                                        out_shapes=self.out_shapes,
                                        **self.template_kwargs)
        return outputs

    def input_name(self, type, input_id):
        """
        push_0_Stack
        """
        return "%s-%s-%s" % (self.name, type.name, input_id)

    def get_params(self, **tags):
        return self.params.get_params(**tags)

    def load_params(self, param_values):
        params = self.params.get_params()
        assert len(param_values) == len(params), "Invalid param file"
        for i in range(len(params)):
            params[i].set_value(param_values[i])

    def load_params_fname(self, fname):
        params_file = np.load(fname)
        param_values = npz_to_array(params_file)
        return self.load_params(param_values)

    def save_params(self, fname, compress=True):
        params = self.params.get_params()
        param_values = [param.get_value() for param in params]
        print("Params", params)
        print("Param Sizes", [p.get_value().shape for p in params])
        print("Before Set Means", [p.get_value().mean() for p in params])
        if compress:
            np.savez_compressed(fname, *param_values)
        else:
            np.savez(fname, *param_values)

    def compile(self):
        print("Compiling func")
        call_fn = function(self.inputs, self.outputs, name=self.name)
        return call_fn


class ForAllVar():
    "Universally quantified variable"
    def __init__(self, type):
        self.type = type
        self.input_var = type.tensor(add_batch=True)


class Axiom():
    def __init__(self, lhs, rhs, name=''):
        assert len(lhs) == len(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def get_losses(self, dist=mse):
        print("lhs", self.lhs)
        print("rhs", self.rhs)
        losses = [dist(self.lhs[i], self.rhs[i]) for i in range(len(self.lhs))]
        return losses


class CondAxiom():
    "If cond_lhs= cond_rhs then conseq_lhs = conseq_rhs else alt_lhs = alt_rhs"
    def __init__(self, cond_lhs, cond_rhs, conseq_lhs, conseq_rhs, alt_lhs,
                 alt_rhs, name=''):
        assert len(cond_lhs) == len(cond_rhs) == len(conseq_lhs) == len(conseq_rhs) == len(alt_lhs) == len(alt_rhs)
        self.cond_lhs = cond_lhs
        self.cond_rhs = cond_rhs
        self.conseq_lhs = conseq_lhs
        self.conseq_rhs = conseq_rhs
        self.alt_lhs = alt_lhs
        self.alt_rhs = alt_rhs
        self.num_constraints = len(cond_lhs)

    def get_losses(self, dist=mse):
        losses = []
        for i in self.num_constraints:
            cond = dist(self.cond_lhs[i], self.cond_rhs)
            conseq = dist(self.conseq_lhs[i], self.conseq_rhs)
            alt = dist(self.alt_lhs[i], self.alt_rhs)
            loss.append(cond*conseq + (1-cond) * alt)
        return losses
        return losses

class BoundAxiom():
    "Constraints a type to be within specifiec bounds"
    def __init__(self, type, name='bound_loss'):
        self.input_var = type

    def get_losses(self):
        return [bound_loss(self.input_var).mean()]


class Const():
    def __init__(self, type, spec, name='C'):
        self.type = type
        self.shape = type.get_shape(add_batch=True, batch_size=1)
        arr = spec(self.shape)
        arr = floatX(arr)
        assert arr.shape == self.shape
        broadcastable = (True,) + (False,) * (len(self.shape) - 1)
        # broadcastable = None
        self.input_var = common.variable(arr, name=name,
                                         broadcastable=broadcastable)

    def get_params(self, **tags):
        return [self.input_var]

    def load_params(self, param_value):
        assert self.shape == param_value.shape
        self.input_var.set_value(param_value)

    def load_params_fname(self, fname):
        params_file = np.load(fname)
        param_values = npz_to_array(params_file)
        return self.load_params(param_values[0])

    def save_params(self, fname, compress=True):
        param_value = self.input_var.get_value()
        if compress:
            np.savez_compressed(fname, param_value)
        else:
            np.savez(fname, param_value)

class Params():
    def __init__(self):
        self.params = {}
        self.is_locked = False

    def lock(self):
        self.is_locked = True

    def check(self, params):
        # FIXME, implement check to see all parameters are there
        return True

    def __getitem__(self, key_default_value):
        key, default_value = key_default_value
        return self.get(key, default_value)

    def get(self, key, default_value):
        if key in self.params:
            # print("Retrieving Key")
            return self.params[key]
        else:
            assert not self.is_locked, "Cant create param when locked"
            # print("Creating new key")
            param = default_value
            self.params[key] = param
            return param

    def set(self, key, value):
        if self.is_locked:
            # print("Not Setting, locked")
            return
        if key in self.params:
            self.params[key] = value
        else:
            print("Setting value before generated")
            exit(1)

    def add_tagged_params(self, tagged_params):
        self.tagged_params = tagged_params

    def get_params(self, **tags):
        result = list(self.tagged_params.keys())
        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.tagged_params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.tagged_params[param] & exclude)]

        return lasagne.utils.collect_shared_vars(result)


class AbstractDataType():
    def __init__(self, funcs, consts, forallvars, axioms, name=''):
        self.funcs = funcs
        self.consts = consts
        self.forallvars = forallvars
        self.axioms = axioms
        self.name = name

    def load_params(self, sfx):
        for i in range(len(self.funcs)):
            self.funcs[i].load_params_fname("%s_interface_%s.npz" % (sfx, i))
        for i in range(len(self.consts)):
            self.consts[i].load_params_fname("%s_constant_%s.npz" % (sfx, i))

    def save_params(self, sfx, compress=True):
        for i in range(len(self.funcs)):
            self.funcs[i].save_params("%s_interface_%s" % (sfx, i), compress)
        for i in range(len(self.consts)):
            self.consts[i].save_params("%s_constant_%s" % (sfx, i), compress)


class ProbDataType():
    """ A probabilistic data type gives a function (space) to each funcs,
        a value to each constant and a random variable to each diti=rbution"""
    def __init__(self, adt, train_fn, call_fns, generators, gen_to_inputs,
                 train_outs):
        self.adt = adt
        self.train_fn = train_fn
        self.call_fns = call_fns
        self.generators = generators
        self.gen_to_inputs = gen_to_inputs
        self.train_outs = train_outs

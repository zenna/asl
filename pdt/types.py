from pdt.config import floatX
from pdt.distances import mse, mae
from pdt.util.backend import variable, repeat_to_batch, placeholder
from pdt.distances import *
import time
import numpy as np
from io import *
import tensorflow as tf
from tensorflow import Tensor

# import theano
# import theano.tensor as T
# import lasagne
# from lasagne.utils import floatX
# from theano common.variable theano import function
# from theano import confi  g


def typed_arg_name(type_name, arg_name):
    return "%s::%s" % (arg_name, type_name)


class Type():
    def __init__(self, shape, name, dtype=floatX):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def tensor(self, name, add_batch=False):
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
    def __init__(self, lhs, rhs, name, template=None, tf_interface=None):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.inp_shapes = [type.get_shape(add_batch=True) for type in lhs]
        self.out_shapes = [type.get_shape(add_batch=True) for type in rhs]

        # Initially false because the first __call__ should gen parameters
        self.reuse = False
        assert not (template is None and tf_interface is None)
        if tf_interface is not None:
            self.tf_interface = tf_interface
        else:
            template_f = template['template']
            def tf_func(inputs):
                output, params = template_f(inputs,
                                            inp_shapes=self.inp_shapes,
                                            out_shapes=self.out_shapes,
                                            reuse=self.reuse,
                                            **template)
                return output
            self.tf_interface = tf_func

    def __call__(self, *raw_args):
        args = [arg.input_var if hasattr(arg, 'input_var') else arg for arg in raw_args]
        print("Calling", args)
        # output_args = {'batch_norm_update_averages' : True, 'batch_norm_use_averages' : False}
        output_args = {'deterministic': True}
        with tf.name_scope(self.name):
            with tf.variable_scope(self.name, reuse=self.reuse) as scope:
                outputs = self.tf_interface(args)

        # And from now on reuse parameters
        self.reuse=True
        return outputs

    def get_params(self, trainable=False):
        "Get the variables associated with this interface"
        if trainable:
            variables = tf.GraphKeys.TRAINABLE_VARIABLES
        else:
            variables = tf.GraphKeys.GLOBAL_VARIABLES
        return tf.get_collection(variables,
                                 scope=self.name)

    def to_python_lambda(self, sess):
        """Generate a callable python function for this interface function"""

        def func(*args, sess=sess):
            assert len(args) == len(self.inputs), "Expected %s inputs, got %s" % (len(self.inputs), len(args))
            feed_dict = dict(zip(self.inputs, args))
            outputs = sess.run(self.outputs, feed_dict=feed_dict)
            return outputs

        func.__doc__ = """"%s : [] -> []""" % self.name # TODO FINISH

        return func

    def input_name(self, type, input_id):
        """
        push_0_Stack
        """
        return "%s-%s-%s" % (self.name, type.name, input_id)


class ForAllVar():
    "Universally quantified variable"
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.input_var = type.tensor(self.forallvar_name(), add_batch=True)

    def forallvar_name(self):
        """
        0_Stack
        """
        return "%s-%s" % (self.name, self.type.name)


class Axiom():
    def __init__(self, lhs, rhs, name, restrict_to=None):
        assert len(lhs) == len(rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.name = name
        self.restrict_to = None if restrict_to is None else restrict_to

    def get_losses(self, dist=mse):
        print("lhs", self.lhs)
        print("rhs", self.rhs)
        losses = [dist(self.lhs[i], self.rhs[i]) for i in range(len(self.lhs))]
        return losses


def gt(a, b):
    return -tf.min(a - b, 0)


class GtAxiom():
    """Greater than axiom"""
    def __init__(self, lhs, rhs, name=''):
        assert len(lhs) == len(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def get_losses(self, dist=mse):
        losses = [gt(self.lhs[i], self.rhs[i]) for i in range(len(self.lhs))]
        return losses


def hard_unit_bound(t):
    return t
    # return tf.minimum(tf.maximum(t, 0.0), 1.0)

def iden(t):
    return t

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

    def get_losses(self, dist=mse, uib=iden):
        losses = []
        for i in range(self.num_constraints):
            cond = uib(dist(self.cond_lhs[i], self.cond_rhs, reduce_batch=False))
            conseq = uib(dist(self.conseq_lhs[i], self.conseq_rhs, reduce_batch=False))
            alt = uib(dist(self.alt_lhs[i], self.alt_rhs, reduce_batch=False))
            coseq_loss = real_and(real_not(cond), conseq)
            alt_loss = real_and(cond, alt)
            either = real_or(coseq_loss, alt_loss, uib=uib)
            # either = tf.Print(either, [self.cond_lhs[i]], message="hello")
            losses.append(tf.reduce_mean(either))
        return losses

def real_or(a, b, uib=tf.nn.sigmoid):
    return uib(a + b)

def real_and(a, b):
    return a * b

def real_not(a):
    return 1-a

# def real_xor(a, b):
#     real_and(real_or(a, b), real_not(real_or())

## Unit Interval Bounds

class BoundAxiom():
    "Constraints a type to be within specifiec bounds"
    def __init__(self, type, name='bound_loss'):
        self.input_var = type

    def get_losses(self):
        return [bound_loss(self.input_var).mean()]


class Loss():
    "A value to be minimized"

    def __init__(self, loss: Tensor,  name: str, restrict_to=None):
        """Create a loss term, only functions and constants in `restrict_to`
        (unless it is None) will be optimized to minimize this loss"""
        self.loss = loss
        self.name = name
        self.restrict_to = None if restrict_to is None else restrict_to


class Const():
    def __init__(self,
                 type: Type,
                 name: str,
                 batch_size: int,
                 initializer,
                 do_repeat_to_batch=True):
        self.type = type
        self.shape = type.get_shape(add_batch=True, batch_size=1)
        self.name = name
        # arr = floatX(arr)
        # assert arr.shape == self.shape
        broadcastable = (True,) + (False,) * (len(self.shape) - 1)
        with tf.name_scope(self.const_name()):
            arr = initializer()(self.shape)
            self.input_var = variable(arr, dtype=type.dtype,
                                      name=self.const_name(),
                                      broadcastable=broadcastable)
            if do_repeat_to_batch:
                self.batch_input_var = repeat_to_batch(self.input_var, batch_size)

    def const_name(self):
        """
        0_Stack
        """
        return "const-%s-%s" % (self.name, self.type.name)


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
    def __init__(self, interfaces, consts, forallvars, axioms, losses, name=''):
        self.interfaces = interfaces
        self.consts = consts
        self.forallvars = forallvars
        self.axioms = axioms
        self.losses = losses
        self.name = name

    def load_params(self, sfx):
        for i in range(len(self.interfaces)):
            self.interfaces[i].load_params_fname("%s_interface_%s.npz" % (sfx, i))
        for i in range(len(self.consts)):
            self.consts[i].load_params_fname("%s_constant_%s.npz" % (sfx, i))

    def save_params(self, sfx, compress=True):
        for i in range(len(self.interfaces)):
            self.interfaces[i].save_params("%s_interface_%s" % (sfx, i), compress)
        for i in range(len(self.consts)):
            self.consts[i].save_params("%s_constant_%s" % (sfx, i), compress)


class ProbDataType():
    """ A probabilistic data type gives a function (space) to each interfaces,
        a value to each constant and a random variable to each diti=rbution"""
    def __init__(self, adt, train_generators, test_generators, gen_to_inputs,
                 train_outs):
        self.adt = adt
        self.train_generators = train_generators
        self.test_generators = test_generators
        self.gen_to_inputs = gen_to_inputs
        self.train_outs = train_outs

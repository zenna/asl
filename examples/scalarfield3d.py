# add exponential decay to weights

# add convergence test

# 3d conv net

# Add visualization

# Change uniform to gaussian

# Add dropout

# Get rid of pdt

from pdt import *
from pdt.train_tf import *
from pdt.types import *
from pdt.adversarial import adversarial_losses
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from wacacore.util.tf import dims_bar_batch
import numpy as np
from common import handle_options
import tensorflow as tf
from tensorflow.contrib import rnn
from voxel_helpers import *

def create_encode(field_shape, n_steps, batch_size):
    n_hidden = product(field_shape)

    def encode_tf(inputs):
        '''
        inputs will be (batch_size, 1000, 3)
        '''
        assert len(inputs) == 1
        voxels = inputs[0]
        voxels = tf.split(voxels, n_steps, 1) #[(batch_size, 100, 3)...] 10 elements ; TODO try later n_steps = 1000
        voxels = [tf.reshape(inp, [batch_size, 300]) for inp in voxels] # [batch_size, 300]
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, voxels, dtype=tf.float32)
        return [outputs[-1]]
    return encode_tf


def add_summary(name, tensor):
    t = tf.reduce_sum(tensor, reduction_indices=dims_bar_batch(tensor))
    tf.summary.histogram(name, t)


def gen_scalar_field_adt(train_data,
                         test_data,
                         options,
                         encode_args={'n_input': 300,  'n_steps': 10},
                         field_shape=(8, 8, 8),
                         voxel_grid_shape=(32, 32, 32),
                         batch_size=64,
                         s_args={},
                         decode_args={}):
    # Types
    sample_space_shape = (10,)

    Field = Type(field_shape, name="Field")
    SampleSpace = Type(sample_space_shape, name="SampleSpace")
    Bool = Type((1,), name="Bool")
    VoxelGrid = Type(voxel_grid_shape, name="voxel_grid")

    # Interfaces
    # A random variable over sample
    interfaces = []
    generator = Interface([SampleSpace], [Field], 'generator', template=s_args)
    interfaces.append(generator)

    discriminator = Interface([Field], [Bool], 'discriminator', template=s_args)
    interfaces.append(discriminator)

    encode_interface = create_encode(field_shape,
                                     encode_args['n_steps'],
                                     batch_size)
    encode = Interface([VoxelGrid], [Field], 'encode',
                       tf_interface=encode_interface)
    interfaces.append(encode)

    decode = Interface([Field], [VoxelGrid], 'decode', template=decode_args)

    # Variables
    forallvars = []

    voxel_grid = ForAllVar(VoxelGrid, "voxel_grid")
    add_summary("voxel_input", voxel_grid.input_var)
    forallvars.append(voxel_grid)

    sample_space = ForAllVar(SampleSpace, "sample_space")
    forallvars.append(sample_space)

    # Axioms
    axioms = []

    # Encode Decode
    (encoded_field, ) = encode(voxel_grid.input_var)
    add_summary("encoded_field", encoded_field)

    (decoded_vox_grid, ) = decode(encoded_field)
    add_summary("decoded_vox_grid", decoded_vox_grid)

    axiom_enc_dec = Axiom((decoded_vox_grid, ), (voxel_grid.input_var, ), 'enc_dec')
    axioms.append(axiom_enc_dec)

    # Other loss terms
    data_sample = encoded_field
    losses = adversarial_losses(sample_space,
                                data_sample,
                                generator,
                                discriminator)

    train_outs = []
    gen_to_inputs = identity

    # Generators
    train_generators = []
    test_generators = []

    voxel_gen = infinite_batches(train_data, batch_size, shuffle=True)
    train_generators.append(voxel_gen)

    # Test
    test_voxel_gen = infinite_batches(test_data, batch_size, shuffle=True)
    test_generators.append(test_voxel_gen)

    sample_space_gen = infinite_samples(np.random.rand,
                                        (batch_size),
                                        sample_space_shape,
                                        add_batch=True)
    train_generators.append(sample_space_gen)

    # Test
    test_sample_space_gen = infinite_samples(np.random.rand,
                                             (batch_size),
                                             sample_space_shape,
                                             add_batch=True)
    test_generators.append(test_sample_space_gen)

    interfaces = [encode, decode, discriminator, generator]
    consts = []
    scalar_field_adt = AbstractDataType(interfaces=interfaces,
                                        consts=consts,
                                        forallvars=forallvars,
                                        axioms=axioms,
                                        losses=losses,
                                        name='scalar_field')

    scalar_field_pbt = ProbDataType(scalar_field_adt,
                                    train_generators,
                                    test_generators,
                                    gen_to_inputs,
                                    train_outs)
    return scalar_field_adt, scalar_field_pbt

# def save_voxels()

def run(options):
    global voxel_grids, adt, pdt, sess
    voxel_grids = model_net_40()
    voxel_grid_shape=voxel_grid_shape[0].shape

    # Steam of Pointss
    limit = 1000
    voxel_grid_shape=(limit, 3)
    voxel_grids = voxel_indices(voxel_grids, limit)
    indices_voxels(voxel_grids)

    test_size = 512
    train_voxel_grids = voxel_grids[0:-test_size]
    test_voxel_grids = voxel_grids[test_size:]

    # train_voxel_grids = voxel_grids[0:128]
    # test_voxel_grids = voxel_grids[0:128]

    # Parameters
    encode_args = {'n_input': 300,  'n_steps': 10}

    # Default params
    field_shape = options['field_shape'] if 'field_shape' in options else (100,)
    adt, pdt = gen_scalar_field_adt(train_voxel_grids,
                                    test_voxel_grids,
                                    options,
                                    voxel_grid_shape=voxel_grid_shape,
                                    s_args=options,
                                    field_shape=field_shape,
                                    encode_args=encode_args,
                                    decode_args=options,
                                    batch_size=options['batch_size'])
    sess = train(adt, pdt, options)

def main(argv):
    print("ARGV", argv)
    options = handle_options('scalar_field', argv)
    options['dirname'] = gen_sfx_key(('adt',), options)
    run(options)


def scalar_field_options():
    options = {'field_shape': (eval, (50,))}
    return options

if __name__ == "__main__":
    main(sys.argv[1:])

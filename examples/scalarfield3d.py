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
import numpy as np
from common import handle_options
import tensorflow as tf
from tensorflow.contrib import rnn

def create_encode(field_shape, n_input, n_steps):
    n_hidden = product(field_shape)

    def encode_tf(inputs):
        '''
        inputs will be (?, 3, 1000)
        '''
        assert len(inputs) == 1
        voxels = inputs[0]

        ## RNN
        voxels = tf.transpose(voxels, [1,0,2])
        voxels = tf.reshape(voxels, [-1, n_input])
        voxels = tf.split(voxels, n_steps, 0)
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, voxels, dtype=tf.float32)
        return [outputs[-1]]

    return encode_tf

from wacacore.util.tf import dims_bar_batch

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
    generator = Interface([SampleSpace], [Field], 'generator', template=s_args)
    discriminator = Interface([Field], [Bool], 'discriminator', template=s_args)
    # encode_interface = create_encode(field_shape, encode_args['n_input'],
    #                                  encode_args['n_steps'])
    # encode = Interface([VoxelGrid], [Field], 'encode',
    #                    tf_interface=encode_interface)
    #
    encode = Interface([VoxelGrid], [Field], 'encode', template=s_args)
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


    sample_space_gen =  infinite_samples(np.random.rand,
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

    funcs = [encode, decode, discriminator, generator]
    consts = []
    scalar_field_adt = AbstractDataType(funcs=funcs,
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

def voxel_indices(voxels, limit):
    """
    Convert voxel data_set (n, 32, 32, 32) to (n, 3, m)
    """
    n,x,y,z = voxels.shape
    output = np.ones((n, 3, limit))
    output = output * -1

    # obtain occupied voxels
    for v in range(len(voxels)):
        voxel = voxels[v]
        x_list, y_list, z_list = np.where(voxel)
        assert len(x_list)==len(y_list)
        assert len(y_list)==len(z_list)

        # fill in output tensor
        npoints = min(limit, len(x_list))
        output[v][0][0:npoints] = x_list[0:npoints]
        output[v][1][0:npoints] = y_list[0:npoints]
        output[v][2][0:npoints] = z_list[0:npoints]
    return output



# assert not (template is None and tf_interface is None)
# if tf_interface is not None:
# else:
#     template_f = template['template']
#     def tf_func(inputs):
#         output, params = template_f(inputs,
#                                     inp_shapes=self.inp_shapes,
#                                     out_shapes=self.out_shapes,
#                                     reuse=self.reuse,
#                                     **template)
#         return output
#     self.tf_interface = tf_func

def run(options):
    global voxel_grids, adt, pdt, sess
    datadir = os.environ['DATADIR']

    # 32 * 32 * 32
    voxels_path = os.path.join(datadir, 'ModelNet40', 'alltrain32.npy')
    voxel_grids = np.load(voxels_path)
    voxel_grid_shape=(32, 32, 32)

    # Steam of Pointss
    # limit = 1000
    # voxel_grid_shape=(3, limit)
    # voxel_grids = voxel_indices(voxel_grids, limit)

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

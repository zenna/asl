from pdt import *
from pdt.train_tf import *
from pdt.types import *
from pdt.adversarial import adversarial_losses
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from wacacore.util.tf import dims_bar_batch
from wacacore.train.callbacks import every_n
import numpy as np
from common import handle_options
import tensorflow as tfff
from tensorflow.contrib import rnn
from voxel_helpers import *
from tflearn.layers import conv_3d, fully_connected, conv, conv_2d
from tflearn.layers.normalization import batch_normalization
import tflearn
from scipy import ndimage

# I'm not sure how to rotate the voxels
# if i did a translation
# so rotation seems to be a little complicated
# what if we did a scaling
# i dont know how to do any of thee o

def conv_2d_layer(input, n_filters, stride):
    return conv_2d(input,
                   n_filters,
                   3,
                   strides=stride,
                   padding='same',
                   activation='elu',
                   bias_init='zeros',
                   scope=None,
                   name='Conv3D')

def conv_3d_layer(t, n_filters, stride):
    return conv_3d(t,
                       n_filters,
                       3,
                       strides=stride,
                       padding='same',
                       activation='elu',
                       bias_init='zeros',
                       scope=None,
                       name='Conv3D')

def conv_transpose_layer(input, n_filters, stride, output_shape):
    return conv.conv_3d_transpose(input,
                       n_filters,
                       3,
                       output_shape = output_shape,
                       strides=stride,
                       padding='same',
                       activation='elu',
                       bias_init='zeros',
                       scope=None,
                       name='Conv3D')

def create_encode(field_shape):
    n_hidden = product(field_shape)
    def encode_conv_3d_net(inputs):
        assert len(inputs) == 1
        voxels = inputs[0]
        voxels = tf.expand_dims(voxels, 4)
        curr_layer = voxels
        layers = []
        curr_layer = conv_3d_layer(curr_layer, 4, 1)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_3d_layer(curr_layer, 16, 2)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_3d_layer(curr_layer, 16, 2)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_3d_layer(curr_layer, 8, 1)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_3d_layer(curr_layer, 4, 2)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = tf.reshape(curr_layer, (-1, ) + field_shape)
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        # final = fully_connected(curr_layer, field_shape)
        return [curr_layer]


    return encode_conv_3d_net

def create_decode(voxel_shape):
    def decode_conv_3d_net(inputs):
        assert len(inputs) == 1
        field = inputs[0]
        layers = []

        curr_layer = field
        curr_layer = tf.reshape(curr_layer, (-1, 4, 4, 4, 4))
        curr_layer = conv_transpose_layer(curr_layer, 8, 2, [8, 8, 8])
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_transpose_layer(curr_layer, 16, 1, [8, 8, 8])
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_transpose_layer(curr_layer, 16, 2, [16, 16, 16])
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_transpose_layer(curr_layer, 4, 2, [32, 32, 32])
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = conv_transpose_layer(curr_layer, 1, 1, [32, 32, 32])
        curr_layer = batch_normalization(curr_layer)
        layers.append(curr_layer)
        curr_layer = tf.reshape(curr_layer, (-1, 32, 32, 32))
        layers.append(curr_layer)
        return [curr_layer]


    return decode_conv_3d_net

def generator_net(inputs):
    sample_space = inputs[0]
    curr_layer = sample_space

    layers = []
    curr_layer = tf.reshape(curr_layer, (-1, 16, 16, 1))
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 1, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = tf.reshape(curr_layer, (-1, 16, 16))
    return [curr_layer]


def rotation_net(inputs):
    field = inputs[0]
    field = tf.reshape(field, (-1, 16, 16, 1))
    rotation = inputs[1]
    rotation = tf.reshape(rotation, (-1, 1, 1, 1))
    rotate_tiled = tf.ones_like(field) * rotation
    curr_layer = tf.concat([field, rotate_tiled], 3)
    layers = []
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 1, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = tf.reshape(curr_layer, (-1, 16, 16))
    return [curr_layer]



def discriminator_net(inputs):
    field = inputs[0]
    curr_layer = field

    layers = []
    curr_layer = tf.reshape(curr_layer, (-1, 16, 16, 1))
    curr_layer = conv_2d_layer(curr_layer, 16, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = fully_connected(curr_layer, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = tflearn.activations.sigmoid(curr_layer)
    return [curr_layer]

#
# def create_encode(field_shape, n_steps, batch_size):
#     n_hidden = product(field_shape)
#
#     def encode_tf(inputs):
#         import pdb; pdb.set_trace()
#         '''
#         inputs will be (batch_size, 1000, 4)
#         '''
#         assert len(inputs) == 1
#         voxels = inputs[0]
#         voxels = tf.split(voxels, n_steps, 1) #[(batch_size, 100, 3)...] 10 elements ; TODO try later n_steps = 1000
#         voxels = [tf.reshape(inp, [batch_size, 4000 // n_steps]) for inp in voxels] # [batch_size, 300]
#         lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#         outputs, states = rnn.static_rnn(lstm_cell, voxels, dtype=tf.float32)
#         return [outputs[-1]]
#     return encode_tf


def add_summary(name, tensor):
    t = tf.reduce_sum(tensor, reduction_indices=dims_bar_batch(tensor))
    tf.summary.histogram(name, t)
    with tf.name_scope("variance"):
        tf.summary.scalar("%s_variances" % name, tf.nn.moments(t, axes=[0])[1])


def gen_scalar_field_adt(train_data,
                         test_data,
                         options,
                         encode_args={'n_steps': 10},
                         field_shape=(8, 8, 8),
                         voxel_grid_shape=(32, 32, 32),
                         batch_size=64,
                         s_args={},
                         decode_args={}):

    extra_fetches = {}

    # Types
    # =====

    # Shape parameters
    sample_space_shape = (16, 16)
    rot_matrix_shape = (1, )

    Field = Type(field_shape, name="Field")
    Rotation = Type(rot_matrix_shape, name="Rotation")
    VoxelGrid = Type(voxel_grid_shape, name="VoxelGrid")
    # SampleSpace = Type(sample_space_shape, name="SampleSpace")
    # Bool = Type((1,), name="Bool")

    # Interfaces
    # ==========

    # A random variable over sample
    # generator = Interface([SampleSpace], [Field], 'generator', tf_interface=generator_net)
    # discriminator = Interface([Field], [Bool], 'discriminator', tf_interface=discriminator_net)
    rotate = Interface([Field, Rotation], [Field], 'rotate', tf_interface=rotation_net)

    # Encode 2
    encode_interface = create_encode(field_shape)
    encode = Interface([VoxelGrid], [Field], 'encode', tf_interface=encode_interface)

    decode_interface = create_decode(field_shape)
    decode = Interface([VoxelGrid], [Field], 'decode',
                       tf_interface=decode_interface)

    interfaces = [encode,
                  decode,
                  rotate]

    # Variables
    # =========
    voxel_grid = ForAllVar(VoxelGrid, "voxel_grid")
    rot_voxel_grid = ForAllVar(VoxelGrid, "rot_voxel_grid")
    rot_matrix = ForAllVar(Rotation, "rotation")
    # sample_space = ForAllVar(SampleSpace, "sample_space")
    add_summary("voxel_input", voxel_grid.input_var)

    forallvars = [voxel_grid,
                  rot_matrix,
                  # sample_space,
                  ]

    # Train Generators
    # ================
    train_voxel_gen = infinite_batches(train_data, batch_size, shuffle=True)
    # sample_space_gen = infinite_samples(np.random.randn,
    #                                     (batch_size),
    #                                     sample_space_shape,
    #                                     add_batch=True)
    train_rot_gen = infinite_samples(lambda *shp: np.random.rand(*shp)*360,
                                     batch_size,
                                     rot_matrix_shape,
                                     add_batch=True)

    def test_train_gen(voxel_gen, rot_gen):
        rot_vgrid = np.zeros((batch_size, 32, 32, 32))
        while True:
            sample_vgrids = next(voxel_gen)
            sample_rot_matrix = next(rot_gen)

            for i, vgrid in enumerate(sample_vgrids):
                rot_vgrid[i] = ndimage.interpolation.rotate(sample_vgrids[i], sample_rot_matrix[i], reshape=False)

            vals = {voxel_grid.input_var: sample_vgrids,
                    rot_matrix.input_var: sample_rot_matrix,
                    rot_voxel_grid.input_var: rot_vgrid}

            yield vals

    train_generators = [test_train_gen(train_voxel_gen, train_rot_gen)]

    # Test Generators
    # ================
    test_generators = train_generators


    # Axioms
    # ======
    axioms = []

    # Encode Decode
    (encoded_field, ) = encode(voxel_grid.input_var)
    add_summary("encoded_field", encoded_field)
    extra_fetches["voxel_grid.input_var"] = voxel_grid.input_var

    (decoded_vox_grid, ) = decode(encoded_field)
    add_summary("decoded_vox_grid", decoded_vox_grid)
    extra_fetches["decoded_vox_grid"] = decoded_vox_grid

    axiom_enc_dec = Axiom((decoded_vox_grid, ),
                          (voxel_grid.input_var, ),
                          'enc_dec')
    axioms.append(axiom_enc_dec)

    # rotation axioms
    (rotated,) = rotate(encoded_field, rot_matrix.input_var)
    (dec_rotated_vgrid, ) = decode(rotated)
    axiom_rotate = Axiom((dec_rotated_vgrid, ), (rot_voxel_grid.input_var, ), 'rotate')
    axioms.append(axiom_rotate)

    tf.summary.image("encoded_field", tf.reshape(encoded_field, (-1, 16, 16, 1)))
    tf.summary.image("rotate_field", tf.reshape(rotated, (-1, 16, 16, 1)))

    # Losses
    # ======
    #
    # # Other loss terms
    # data_sample = encoded_field
    # losses, adversarial_fetches = adversarial_losses(sample_space,
    #                             data_sample,
    #                             generator,
    #                             discriminator)
    #
    # # Make the encoder help the generator!!
    # losses[0].restrict_to.append(encode)
    # (generated_voxels, ) = decode(adversarial_fetches['generated_field'])
    # extra_fetches['generated_voxels'] = generated_voxels
    # extra_fetches.update(adversarial_fetches)
    #
    # add_summary("generated_field", adversarial_fetches['generated_field'])
    # add_summary("generated_voxels", generated_voxels)
    losses = []



    # Constants
    # =========
    consts = []

    # Data Types
    # ==========
    scalar_field_adt = AbstractDataType(interfaces=interfaces,
                                        consts=consts,
                                        forallvars=forallvars,
                                        axioms=axioms,
                                        losses=losses,
                                        name='scalar_field')

    scalar_field_pbt = ProbDataType(adt=scalar_field_adt,
                                    train_generators=train_generators,
                                    test_generators=test_generators,
                                    train_outs=[])
    return scalar_field_adt, scalar_field_pbt, extra_fetches


def save_voxels(fetch_data,
               feed_dict,
               i: int,
               **kwargs):
    """Save Voxels"""
    # import pdb; pdb.set_trace()
    return True
    # import pdb; pdb.set_trace()


def run(options):
    global voxel_grids, adt, pdt, sess
    voxel_grids = model_net_40()
    voxel_grid_shape = voxel_grids.shape[1:]

    # Steam of Pointss
    # limit = 1000
    # voxel_grid_shape = (limit, 4)
    # voxel_grids = voxel_indices(voxel_grids, limit)
    # indices_voxels(voxel_grids)

    test_size = 512
    train_voxel_grids = voxel_grids[0:-test_size]
    test_voxel_grids = voxel_grids[test_size:]


    # Parameters
    encode_args = {'n_steps': 100}

    # Default params
    field_shape = options['field_shape'] if 'field_shape' in options else (16, 16)
    adt, pdt, extra_fetches = gen_scalar_field_adt(train_voxel_grids,
                                    test_voxel_grids,
                                    options,
                                    voxel_grid_shape=voxel_grid_shape,
                                    s_args=options,
                                    field_shape=field_shape,
                                    encode_args=encode_args,
                                    decode_args=options,
                                    batch_size=options['batch_size'])

    sess = train(adt, pdt, options, extra_fetches=extra_fetches, callbacks = [every_n(save_voxels, 100)])

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

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

def create_encode(field_shape, n_steps, batch_size):
    n_hidden = product(field_shape)

    def encode_tf(inputs):
        '''
        inputs will be (batch_size, 1000, 3)
        '''
        assert len(inputs) == 1
        voxels = inputs[0]

        ## RNN
        # voxels = tf.transpose(voxels, [1,0,2])
        # pdb.set_trace()

        voxels = tf.split(voxels, n_steps, 1) #[(batch_size, 100, 3)...] 10 elements ; TODO try later n_steps = 1000
        
        voxels = [tf.reshape(inp, [batch_size, 300]) for inp in voxels] # [batch_size, 300] 

        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, voxels, dtype=tf.float32)
        return [outputs[-1]]

    return encode_tf


def rand_rotation_matrix(deflection=1.0, randnums=None, floatX='float32'):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1,
    competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be
    auto-generated.
    """
    # from realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return np.array(M, dtype=floatX)


def euclidean_norm(t, ri):
    with tf.name_scope("euclidean_norm"):
        sqr = tf.square(t)
        norm = tf.reduce_sum(sqr, reduction_indices=ri)
    return norm


def sdf_sphere(t):
    length = euclidean_norm(t, 2)
    return length - 1.2


def unsign(t):
    """Convert a signed distance into 0s at negatives"""
    return tf.nn.relu(t)


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
    funcs = []

    # A random variable over sample
    generator = Interface([SampleSpace], [Field], 'generator', template=s_args)
    funcs.append(generator)

    descriminator = Interface([Field], [Bool], 'descriminator', template=s_args)
    funcs.append(descriminator)

    encode_interface = create_encode(field_shape,
                                     encode_args['n_steps'],
                                     batch_size)
    encode = Interface([VoxelGrid], [Field], 'encode',
                       tf_interface=encode_interface)
    funcs.append(encode)

    decode = Interface([Field], [VoxelGrid], 'decode', template=decode_args)
    funcs.append(decode)

    # Constants
    consts = []

    # Variables
    forallvars = []

    voxel_grid = ForAllVar(VoxelGrid, "voxel_grid")
    forallvars.append(voxel_grid)

    sample_space = ForAllVar(SampleSpace, "sample_space")
    forallvars.append(sample_space)

    # Axioms
    axioms = []

    # Encode Decode
    (encoded_field, ) = encode(voxel_grid.input_var)
    (decoded_vox_grid, ) = decode(encoded_field)
    axiom_enc_dec = Axiom((decoded_vox_grid, ), (voxel_grid.input_var, ), 'enc_dec')
    axioms.append(axiom_enc_dec)

    # Other loss terms
    data_sample = encoded_field
    losses = adversarial_losses(sample_space,
                                data_sample,
                                generator,
                                descriminator)

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

    output = np.transpose(output, [0,2,1]) # switch limit and coords

    return output


def run(options):
    global voxel_grids, adt, pdt, sess
    datadir = os.environ['DATADIR']
    voxels_path = os.path.join(datadir, 'ModelNet40', 'alltrain32.npy')
    voxel_grids = np.load(voxels_path)
    limit = 1000
    voxel_grids = voxel_indices(voxel_grids, limit)

    test_size = 512
    train_voxel_grids = voxel_grids[0:-test_size]
    test_voxel_grids = voxel_grids[test_size:]

    # Parameters
    encode_args = {'n_input': 300,  'n_steps': 10}

    # Default params
    field_shape = options['field_shape'] if 'field_shape' in options else (100,)
    adt, pdt = gen_scalar_field_adt(train_voxel_grids,
                                    test_voxel_grids,
                                    options,
                                    voxel_grid_shape=(limit, 3),
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

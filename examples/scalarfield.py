import numpy as np

from pdt import *
from mnist import *
# from ig.util import *
from common import handle_options, load_train_save
from pdt.train_tf import *
from pdt.common import *
from wacacore.util.io import *
from pdt.types import *
from wacacore.util.generators import infinite_samples, infinite_batches



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


def scalar_field_adt(options, field_shape=(100,),
                     npoints=100, batch_size=64, s_args={}, add_args={},
                     field_args={}, translate_args={}):
    # Types
    # rot_matrix_shape = (3, 3)
    points_shape = (npoints, 3)
    Field = Type(field_shape, name="field")
    Points = Type(points_shape, name="points")
    Scalar = Type((npoints,), name="scalar")
    # Rotation = Type(rot_matrix_shape, name="rotation")

    translate_shape = (3,)
    Translation = Type(translate_shape, name="translate")

    # Interface
    s = Interface([Field, Points], [Scalar], 's', **s_args)
    # add = Interface([Field, Points], [Field], 'add', **add_args)
    # rotate = Interface([Field, Rotation], [Field], 'rotate', **rotate_args)
    translate = Interface([Field, Translation], [Field], 'translate', **translate_args)
    # funcs = [s, rotate]
    funcs = [s, translate]

    # Constants
    sphere_field = Const(Field, "sphere_field", batch_size, **field_args)
    consts = [sphere_field]

    # Variables
    # rot_matrix = ForAllVar(Rotation, "rot_matrix")
    translate_vec = ForAllVar(Translation, "translate_vec")
    pos = ForAllVar(Points, "pos")
    forallvars = [pos, translate_vec]

    # PreProcessing
    sphere_field_batch = sphere_field.batch_input_var

    # Axioms
    # Zero field is zero everywhere
    # axiom1 = Axiom(s(zero_field_batch, poss[0]), (0,))
    # axiom2 = Axiom(s(zero_field_batch, poss[1]), (0,))
    # axioms.append(axiom1)
    # axioms.append(axiom2)

    ## if points is within unit sphere then should be on, otherwise off
    # poses = s(sphere_field_batch)
    is_unit_sphere = unsign(sdf_sphere(pos.input_var))
    sphere_axiom = CondAxiom((is_unit_sphere,), (0.0,),
                             s(sphere_field_batch, pos.input_var), (1.0,),
                             s(sphere_field_batch, pos.input_var), (0.0,))

    # Rotation axioms
    # (rotated,) = rotate(sphere_field_batch, rot_matrix.input_var)
    (translated,) = translate(sphere_field_batch, translate_vec.input_var)
    reshape_translate_vec = tf.reshape(translate_vec.input_var, [batch_size, 1, 3])
    # import pdb; pdb.set_trace()
    axiom_r1 = Axiom(s(translated, pos.input_var),
                     s(sphere_field_batch, pos.input_var + reshape_translate_vec))

    axioms = [sphere_axiom]

    train_outs = []
    gen_to_inputs = identity

    # Generators
    pos_gen = infinite_samples(lambda *x: np.random.rand(*x), batch_size, points_shape)
    # rot_gen = infinite_samples(lambda *x: rand_rotation_matrix(), batch_size, rot_matrix_shape)
    tran_gen = infinite_samples(lambda *x: np.random.rand(*x), batch_size, translate_shape)
    generators = [pos_gen, tran_gen]

    train_fn, call_fns = None, None
    #compile_fns(funcs, consts, forallvars, axioms, train_outs, options)
    scalar_field_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                        name='scalar_field')
    scalar_field_pbt = ProbDataType(scalar_field_adt, train_fn, call_fns,
                                    generators, gen_to_inputs, train_outs)
    return scalar_field_adt, scalar_field_pbt


def run(options):
    # voxel_grids = np.load("/DATADIR/data/ModelNet40/alltrain32.npy")
    field_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = scalar_field_adt(options, s_args=options, translate_args=options,
                                npoints=500, field_shape=(102,),
                                add_args=options, field_args=field_args,
                                batch_size=options['batch_size'])

    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)
    graph = tf.get_default_graph()

    savedir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, savedir)

def main():
    options = handle_options('scalar_field', argv)
    run(options)

if __name__ == "__main__":
   main(sys.argv[1:])

import theano
import numpy as np

from pdt import *
from mnist import *
# from ig.util import *
from pdt.train import *
from pdt.common import *
from pdt.io import *
from pdt.types import *

theano.config.optimizer = 'fast_compile'

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

def scalar_field_adt(options, field_shape=(100,),
                     npoints=100, batch_size=64, s_args={}, add_args={},
                     rotate_args={}):
    # Types
    rot_matrix_shape = (3, 3)
    points_shape = (npoints, 3)
    Field = Type(field_shape)
    Points = Type(points_shape)
    Scalar = Type((npoints,))
    Rotation = Type(rot_matrix_shape)

    # Interface
    s = Interface([Field, Points], [Scalar], 's', **s_args)
    # add = Interface([Field, Points], [Field], 'add', **add_args)
    rotate = Interface([Field, Rotation], [Field], 'rotate', **rotate_args)
    funcs = [s, rotate]

    # Constants
    cube_field = Const(Field)
    consts = [cube_field]

    # Variables
    rot_matrix = ForAllVar(Rotation)
    pos = ForAllVar(Points)
    forallvars = [pos, rot_matrix]

    # PreProcessing
    cube_field_batch = repeat_to_batch(cube_field.input_var, batch_size)

    # Axioms
    # Zero field is zero everywhere
    # axiom1 = Axiom(s(zero_field_batch, poss[0]), (0,))
    # axiom2 = Axiom(s(zero_field_batch, poss[1]), (0,))
    # axioms.append(axiom1)
    # axioms.append(axiom2)

    ## if points is within unit cube then should be on, otherwise off
    # s(cube_field_batch)
    # cube_axiom = CondAxiom(in_unit_cube(poss[1].input_var), true, s(pos), (1), s(pos), (0,))

    # Rotation axioms
    (rotated,) = rotate(cube_field_batch, rot_matrix.input_var)
    axiom_r1 = Axiom(s(rotated, pos.input_var),
                     s(cube_field_batch, pos.input_var*rot_matrix.input_var))

    axioms = [axiom_r1]

    train_outs = []
    gen_to_inputs = identity

    # Generators
    pos_gen = infinite_samples(lambda *x: np.random.rand(*x), batch_size, points_shape)
    rot_gen = infinite_samples(lambda *x: rand_rotation_matrix(), batch_size, rot_matrix_shape)
    generators = [pos_gen, rot_gen]

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms, train_outs, options)
    scalar_field_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                        name='scalar field')
    scalar_field_pbt = ProbDataType(scalar_field_adt, train_fn, call_fns,
                                    generators, gen_to_inputs, train_outs)
    return scalar_field_adt, scalar_field_pbt


def main(argv):
    # Args
    global options
    global test_files, train_files
    global views, outputs, net
    global push, pop
    global X_train
    global adt, pdt
    global sfx
    global save_dir

    cust_options = {}
    cust_options['train'] = (True,)
    cust_options['batch_size'] = (int, 512)
    cust_options['width'] = (int, 28)
    cust_options['height'] = (int, 28)
    cust_options['num_epochs'] = (int, 100)
    cust_options['save_every'] = (int, 100)
    cust_options['compress'] = (False,)
    cust_options['compile_fns'] = (True,)
    cust_options['save_params'] = (True,)
    cust_options['adt'] = (str, 'scalar_field')
    cust_options['template'] = (str, 'res_net')
    cust_options.update(res_net_kwargs())
    options = handle_args(argv, cust_options)

    # voxel_grids = np.load("/home/zenna/data/ModelNet40/alltrain32.npy")
    options['template'] = parse_template(options['template'])
    adt, pbt = scalar_field_adt(options, s_args=options, rotate_args=options,
                                npoints=500, field_shape=(102,),
                                add_args=options,
                                batch_size=options['batch_size'])

    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)

    save_dir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
   main(sys.argv[1:])

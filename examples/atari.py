"Learning a generative model for atari games"
from pdt import *
from pdt.train_tf import *
from pdt.types import *
from pdt.adversarial import adversarial_losses
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
import numpy as np
from common import handle_options
import gym
import random 

## How to do atari games.
# Generate an image

def generate_atari_image_batch(batch_size, game):
    """
    Args:
        batch_size:
        game: name of game, like 'Breakout-v0'
    Return:
        Tensor of size (batch_size, screen_height, screen_width, channels, n)
        for i = 1:batch_size:
            run a game with `commands` and capture images
    """
    env = gym.make(game)
    actions = env.env.get_action_meanings() # list of actions
      # action index corresponds to numerical actions in env.step

    # Output tensor dimensions
    num_actions = env.action_space.n
    screen_height = env.env.ale.getScreenDims()[1]
    screen_width = env.env.ale.getScreenDims()[0]
    output = np.zeros((batch_size, num_actions+1, screen_height, screen_width, 3))

    # Get image data
    for i in range(batch_size):
      env.seed(random.randint(0,10e9)) #TODO: Might want to get data from later stages of game
      env.reset()
      screen = env.env.ale.getScreenRGB()[:, :, :3] # Initial screen
      output[i][0] = screen

      # Save screen after each action
      #TODO: Select random action
      for j in range(num_actions):
        env.step(j)
        screen = env.env.ale.getScreenRGB()[:, :, :3]
        output[i][j+1] = screen
        
    return output

def gen_atari_adt(train_data,
                  test_data,
                  options,
                  field_shape=(8, 8, 8),
                  voxel_grid_shape=(32, 32, 32),
                  npoints=100,
                  batch_size=64,
                  s_args={},
                  encode_args={},
                  decode_args={},
                  add_args={},
                  field_args={},
                 translate_args={}):
     # Types
    sample_space_shape = (10,)
    State = Type(state_shape, name="State")
    Image = Type(image_shape, name="Image")
    SampleSpace = Type(sample_space_shape, name="SampleSpace")

    # Interfaces
    funcs = []

    # A random variable over sample
    render = Interface([State], [Image], 'render', **render_args)
    funcs.append(render)

    up = Interface([State], [State], 'up', **button_args)
    down = Interface([State], [State], 'down', **button_args)
    left = Interface([State], [State], 'left', **button_args)
    right = Interface([State], [State], 'right', **button_args)
    return atari_adt, atari_pbt


def run(options):
    global voxel_grids, adt, pdt, sess

    field_args = {'initializer': tf.random_uniform_initializer}

    # Default params
    npoints = options['npoints'] if 'npoints' in options else 500
    field_shape = options['field_shape'] if 'field_shape' in options else (100,)
    adt, pdt = gen_atari_adt(options,
                             s_args=options,
                             translate_args=options,
                             npoints=500,
                             field_shape=field_shape,
                             encode_args=options,
                             decode_args=options,
                             add_args=options,
                             field_args=field_args,
                             batch_size=options['batch_size'])
    sess = train(adt, pdt, options)

def main(argv):
    print("ARGV", argv)
    options = handle_options('atari', argv)
    options['dirname'] = gen_sfx_key(('adt',), options)
    run(options)


def atari_options():
    options = {'field_shape': (eval, (50,))}
    return options

if __name__ == "__main__":
    main(sys.argv[1:])

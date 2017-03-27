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
from tflearn.layers import conv_3d, fully_connected, conv, conv_2d
from tflearn.layers.normalization import batch_normalization
import tflearn


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

def conv_2d_transpose_layer(input, n_filters, stride, output_shape):
    return conv.conv_2d_transpose(input,
                                  n_filters,
                                  3,
                                  output_shape = output_shape,
                                  strides=stride,
                                  padding='same',
                                  activation='elu',
                                  bias_init='zeros',
                                  scope=None,
                               name='Conv3D')


# Interfaces
def render_tf(inputs):
    state = inputs[0]
    curr_layer = state
    layers = []
    curr_layer = conv_2d_transpose_layer(curr_layer, 9, 1, [42, 32, 9])
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer = conv_2d_transpose_layer(curr_layer, 9, 5, [210, 160, 9])
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer =  conv_2d_transpose_layer(curr_layer, 9, 1, [210, 160, 9])
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer = conv_2d_transpose_layer(curr_layer, 3, 1, [210, 160, 3])
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)
    return [curr_layer]


def inv_render_tf(inputs):
    image = inputs[0]
    curr_layer = image
    layers = []
    curr_layer = conv_2d_layer(curr_layer, 9, 1)
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer = conv_2d_layer(curr_layer, 9, 1)
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer = conv_2d_layer(curr_layer, 9, 5)
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)

    curr_layer = conv_2d_layer(curr_layer, 1, 1)
    curr_layer = batch_normalization(curr_layer)
    layers.append(curr_layer)
    return [curr_layer]


def button(inputs):
    state = inputs[0]
    curr_layer = state

    layers = []
    curr_layer = conv_2d_layer(curr_layer, 9, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 1, 1)
    curr_layer = batch_normalization(curr_layer)
    return [curr_layer]


# 1. Randomize the startpoint of the batch
# 1. Have another thread fill out a buffer for the batch

def gen_atari_data(actions, batch_size):
    env = gym.make('Breakout-v0')
    action_meanings = env.env.get_action_meanings() # list of actionsbb
    action_indices = [action_meanings.index(action) for action in actions]
    assert all((i >= 0 for i in action_indices))

    while True:
        yield generate_atari_image_batch(env, action_indices, batch_size)

def generate_atari_image_batch(env, actions, batch_size):
    """
    Args:
        batch_size:
        game: name of game, like 'Breakout-v0'
    Return:
        Tensor of size (batch_size, screen_height, screen_width, channels, n)
        for i = 1:batch_size:
            run a game with `actions` and capture images
    """
    # Output tensor dimensions
    num_actions = len(actions)
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
      for j, action in enumerate(actions):
        env.step(action)
        screen = env.env.ale.getScreenRGB()[:, :, :3]
        output[i][j+1] = screen

    return output

def gen_atari_adt(batch_size,
                  action_seq = ['LEFT', 'RIGHT', 'FIRE', 'NOOP']):
    # A state represents internal state of the world
    state_shape = (42, 32, 1)
    State = Type(state_shape, name="State")

    # An image is what we see on the screen
    image_shape = (210, 160, 3)
    Image = Type(image_shape, name="Image")

    # Interfaces
    interfaces = []

    render = Interface([State], [Image], 'render_tf', tf_interface=render_tf)
    inv_render = Interface([Image], [State], 'inv_render', tf_interface=inv_render_tf)

    # One interface for every aciton
    left = Interface([State], [State], 'LEFT', tf_interface=button)
    right = Interface([State], [State], 'RIGHT', tf_interface=button)
    fire = Interface([State], [State], 'FIRE', tf_interface=button)
    no_op = Interface([State], [State], 'NOOP', tf_interface=button)
    interfaces = [left, right, fire, no_op, render, inv_render]
    name_to_action = {i.name: i for i in interfaces}

    # The only observable data is image data
    num_actions = len(action_seq)

    # We'll create one variable for each image corresponding to an action
    img_data = [ForAllVar(Image, "image_{}".format(i)) for i in range(num_actions+1)]
    def split_images_by_action(gen_data):
        return [gen_data[:, i, :, :, :] for i in range(gen_data.shape[1])]

    curr_image = img_data[0].input_var
    (curr_state,) = inv_render(curr_image)
    intermediate_states = []
    intermediate_images = []

    axioms = []

    # Execute the
    for i in range(num_actions+1):
        intermediate_states.append(curr_state)
        (curr_state_img_guess, ) = render(curr_state)
        intermediate_images.append(curr_state_img_guess)

        curr_state_img = img_data[i].input_var
        # (curr_state_guess, ) = inv_render(curr_img)
        axiom = Axiom((curr_state_img_guess, ),
                      (curr_state_img, ),
                      "img_{}_is_same".format(i))
        axioms.append(axiom)

        # There are only n actions, but n+1 images, so skip action on last step
        if i < num_actions:
            (curr_state, ) = name_to_action[action_seq[i]](curr_state)

    train_generators = [gen_atari_data(action_seq, batch_size=batch_size)]
    test_generators = [gen_atari_data(action_seq, batch_size=batch_size)]

    forallvars = img_data
    atari_adt = AbstractDataType(interfaces=interfaces,
                                 consts=[],
                                 forallvars=forallvars,
                                 axioms=axioms,
                                 losses=[],
                                 name='atari')

    extra_fetches = {}
    gen_to_inputs = split_images_by_action
    train_outs = []
    atari_pbt = ProbDataType(atari_adt,
                             train_generators,
                             test_generators,
                             gen_to_inputs,
                             train_outs)

    return atari_adt, atari_pbt, extra_fetches

def run(options):
    adt, pdt, extra_fetches = gen_atari_adt(batch_size=options['batch_size'])
    sess = train(adt,
                 pdt,
                 options,
                 extra_fetches=extra_fetches,
                 callbacks=[])

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

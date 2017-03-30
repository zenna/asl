"Learning a generative model for atari games"
from pdt import *
from pdt.train_tf import *
from pdt.types import *
from pdt.adversarial import adversarial_losses
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from tflearn.layers import conv_3d, fully_connected, conv, conv_2d
from tflearn.layers.normalization import batch_normalization
from common import handle_options
import numpy as np
import gym
import random
import tflearn
import pickle
# import pdb

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


def score_net(inputs):
    state = inputs[0]
    curr_layer = state
    layers = []
    curr_layer = conv_2d_layer(curr_layer, 8, 1)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = conv_2d_layer(curr_layer, 2, 4)
    curr_layer = batch_normalization(curr_layer)
    curr_layer = fully_connected(curr_layer, 1, activation='elu')
    curr_layer = batch_normalization(curr_layer)
    return [curr_layer]


# 1. Randomize the startpoint of the batch
# 1. Have another thread fill out a buffer for the batch

def gen_atari_data(env, actions, batch_size):

    action_meanings = env.env.get_action_meanings() # list of actions
    action_indices = [action_meanings.index(action) for action in actions]
    assert all((i >= 0 for i in action_indices))

    while True:
        yield generate_atari_image_batch(env, action_indices, batch_size)

def generate_atari_image_batch(env, actions, batch_size): #TODO: remove env from here, and get from pickle
    """
    Args:
        env: game environment
        actions: list of numerical actions
        batch_size: batches of data

    Return:
        Tensor of size (batch_size, screen_height, screen_width, channels, n)
        for i = 1:batch_size:
            run a game with `actions` and capture images
    """
    # Output tensor dimensions
    # (env, states) = pickle.load(open('examples/states.p', 'rb'))
    num_actions = len(actions)
    screen_height = env.env.ale.getScreenDims()[1]
    screen_width = env.env.ale.getScreenDims()[0]
    output = np.zeros((batch_size, num_actions+1, screen_height, screen_width, 3))

    # Get image data
    rewards = np.zeros((batch_size, num_actions))
    for i in range(batch_size):
      # state = random.choice(states)
      # env.env.ale.restoreState(state) # Select a random state to revisit
      # env.seed(random.randint(0,10e9))
      # env.reset()
      screen = env.env.ale.getScreenRGB()[:, :, :3] # Initial screen
      output[i][0] = screen

      # Save screen after each action
      for j, action in enumerate(actions):
        observation, reward, done, info = env.step(action)
        screen = env.env.ale.getScreenRGB()[:, :, :3]
        output[i][j+1] = screen
        rewards[i, j] = reward

    return output, rewards

def gen_atari_adt(env,
                  batch_size): # used to have action seq as input
    ignored_actions = ['DOWNRIGHT','DOWNLEFT', 'UPRIGHT', 'UPLEFT',
        'UPRIGHTFIRE','UPLEFTFIRE', 'DOWNRIGHTFIRE','DOWNLEFTFIRE']
    action_seq = [action for action in env.env.get_action_meanings() if action not in ignored_actions]
    print("\nACTION SEQ. : ", action_seq)

    # A state represents internal state of the world
    state_shape = (42, 32, 1)
    State = Type(state_shape, name="State")

    # An image is what we see on the screen
    image_shape = (210, 160, 3)
    Image = Type(image_shape, name="Image")

    # A score is the game score
    # pdb.set_trace()
    score_shape = (1,)
    Score = Type(score_shape, name="Score")

    # Interfaces
    interfaces = []

    render = Interface([State], [Image], 'render_tf', tf_interface=render_tf)
    inv_render = Interface([Image], [State], 'inv_render', tf_interface=inv_render_tf)
    score = Interface([State], [Score], "score", tf_interface=score_net)


    # One interface for every action
    interfaces = [Interface([State], [State], action, tf_interface=button) for action in action_seq]
    name_to_action = {action_seq[i].lower(): interfaces[i] for i in range(len(interfaces))}

    interfaces.append(render)
    interfaces.append(inv_render)
    interfaces.append(score)

    name_to_action['render'] = render
    name_to_action['inv_render'] = inv_render
    name_to_action['score'] = score

    # The only observable data is image data
    num_actions = len(action_seq)

    # We'll create one variable for each image and score corresponding to an action
    img_vars = [ForAllVar(Image, "image_{}".format(i)) for i in range(num_actions+1)]
    score_data = [ForAllVar(Score, "score_{}".format(i)) for i in range(num_actions)]

    curr_image = img_vars[0].input_var
    (curr_state,) = inv_render(curr_image)
    intermediate_states = []
    intermediate_images = []

    axioms = []

    # Execute the
    for i in range(num_actions+1):
        intermediate_states.append(curr_state)

        (curr_state_img_guess, ) = render(curr_state)
        intermediate_images.append(curr_state_img_guess)

        curr_state_img = img_vars[i].input_var
        # (curr_state_guess, ) = inv_render(curr_img)

        axiom = Axiom((curr_state_img_guess, ),
                      (curr_state_img, ),
                      "img_{}_is_same".format(i))
        axioms.append(axiom)


        acto = ["BEGIN"] + action_seq
        with tf.name_scope("Guesses"):
            tf.summary.image("img_guess_{}_{}".format(i, acto[i]), curr_state_img_guess)
        with tf.name_scope("Real"):
            tf.summary.image("real_{}_{}".format(i, acto[i]), curr_state_img)
        with tf.name_scope("State"):
            tf.summary.image("state_{}_{}".format(i, acto[i]), curr_state)


        # There are only n actions, but n+1 images, so skip action on last step
        if i < num_actions:
            (curr_state, ) = name_to_action[action_seq[i].lower()](curr_state)
            curr_score = score_data[i].input_var
            (curr_score_guess, ) = score(curr_state)
            axiom_score = Axiom((curr_score_guess, ),
                                (curr_score, ),
                                "score_{}_is_same".format(i))
            axioms.append(axiom_score)

    # Generators
    # ==========
    # Test
    train_generator = gen_atari_data(env, action_seq, batch_size=batch_size)
    test_generator = gen_atari_data(env, action_seq, batch_size=batch_size)

    def split_images_by_action(gen_data):
        return [gen_data[:, i, :, :, :] for i in range(gen_data.shape[1])]

    def make_gen(generator):
        while True:
            sample_img_data, rewards = next(generator)
            img_slices = split_images_by_action(sample_img_data)
            img_feed_dict = {img_vars[i].input_var: img_slices[i] for i in range(num_actions+1)}
            rewards_feed_dict = {score_data[i].input_var: rewards[:, i:i+1] for i in range(num_actions)}
            feed_dict = {}
            feed_dict.update(img_feed_dict)
            feed_dict.update(rewards_feed_dict)
            yield feed_dict

    train_generators = [make_gen(train_generator)]
    test_generators = [make_gen(test_generator)]

    forallvars = img_vars
    atari_adt = AbstractDataType(interfaces=interfaces,
                                 consts=[],
                                 forallvars=forallvars,
                                 axioms=axioms,
                                 losses=[],
                                 name='atari')

    extra_fetches = {}
    train_outs = []
    atari_pbt = ProbDataType(atari_adt,
                             train_generators,
                             test_generators,
                             train_outs)

    return atari_adt, atari_pbt, extra_fetches

def run(options):
    env = gym.make('Breakout-v0')
    env.reset()
    adt, pdt, extra_fetches = gen_atari_adt(env, batch_size=options['batch_size'])
    sess = train(adt,
                 pdt,
                 options,
                 extra_fetches=extra_fetches,
                 callbacks=[])
    return adt, sess

def main(argv):
    print("ARGV", argv)
    options = handle_options('atari', argv)
    options['dirname'] = gen_sfx_key(('adt',), options)
    adt, sess = run(options)
    play(adt.interfaces, adt.consts, sess)


def atari_options():
    options = {'field_shape': (eval, (50,))}
    return options

import matplotlib.pyplot as plt
plt.ion

def play(interfaces, constants, sess):
    interface = {f.name:f for f in interfaces}
    py_interfaces = {f.name:f.to_python_lambda(sess) for f in interfaces}
    action_seq = ['LEFT', 'RIGHT', 'FIRE', 'NOOP']
    data_gen = gen_atari_data(['LEFT', 'RIGHT', 'FIRE', 'NOOP'], 1)
    init_data = next(data_gen)
    init_screen = init_data[0:1, 0]
    init_state = py_interfaces['inv_render'](init_screen)
    state = init_state

    plt.figure()

    # Render the gam
    while True:
        action = np.random.choice(action_seq)
        state = py_interfaces[action](state)
        image = py_interfaces['render'](state)
        plt.imshow(image)



if __name__ == "__main__":
    main(sys.argv[1:])

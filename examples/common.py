import sys
from pdt.common import *
from pdt.util.io import *
from pdt.train_tf import *


def load_train_save(options, adt, pbt, sfx, save_dir):
    options_path = os.path.join(save_dir, "options")
    save_dict_csv(options_path, options)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if options['load_params'] is True:
        saver.restore(sess, options['params_file'])
        # adt.load_params(options['params_file'])

    # if options['save_params'] is True:
    #     path = os.path.join(save_dir, "final" + sfx)
    #     # adt.save_params(path)

    if options['train'] is True:
        train(adt, pbt, sess, num_epochs=options['num_epochs'],
              sfx=sfx, save_dir=save_dir, save_every=options['save_every'],
              compress=options['compress'], saver=saver)

    return sess
# 
# def load_train_save2(options, adt, pbt, sfx, save_dir):
#     options_path = os.path.join(save_dir, "options")
#     save_dict_csv(options_path, options)
#     sess = tf.Session()
#     sess.run(tf.initialize_all_variables())
#     saver = tf.train.Saver()
#
#     if options['load_params'] is True:
#         saver.restore(sess, options['params_file'])
#         # adt.load_params(options['params_file'])
#
#     # if options['save_params'] is True:
#     #     path = os.path.join(save_dir, "final" + sfx)
#     #     # adt.save_params(path)
#
#     if options['train'] is True:
#
#     train(adt, pbt, sess, options)
#
#     return sess

def boolify(x):
    if x in ['0', 0, False, 'False', 'false']:
        return False
    elif x in ['1', 1, True, 'True', 'true']:
        return True
    else:
        assert False, "couldn't convert to bool"

def handle_options(adt, argv):
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options['train'] = (boolify, 1)
    options['nitems'] = (int, 3)
    options['width'] = (int, 28)
    options['height'] = (int, 28)
    options['num_epochs'] = (int, 100)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['compress'] = (boolify, 0)
    options['compile_fns'] = (boolify, 1)
    options['save_params'] = (boolify, 1)
    options['adt'] = (str, adt)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    return options

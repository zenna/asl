import tensorflow as tf


"Generate adversarial loss terms"
def adversarial_losses(sample_space, data_sample, generator, descriminator):
    (gen_sample, ) = generator(sample_space)
    (fake_class, ) = descriminator(data_sample)
    (real_class, ) = descriminator(gen_sample)
    loss_d = tf.reduce_mean(-tf.log(fake_class) - tf.log(1 - real_class))
    loss_g = tf.reduce_mean(-tf.log(real_class))
    return {'generator_loss' : loss_g,
            'descriminator_loss': loss_d}

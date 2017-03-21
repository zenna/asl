import tensorflow as tf
from pdt.types import Loss

"Generate adversarial loss terms"
def adversarial_losses(sample_space,
                       data_sample,
                       generator,
                       descriminator):
    with tf.name_scope("GAN"):
        loss_restrictions = {}
        (gen_sample, ) = generator(sample_space)
        (fake_class, ) = descriminator(data_sample)
        (real_class, ) = descriminator(gen_sample)
        sig_fake_class = tf.sigmoid(fake_class)
        sig_real_class = tf.sigmoid(real_class)
        loss_d = tf.reduce_mean(-tf.log(sig_fake_class) - tf.log(1 - sig_real_class))
        loss_g = tf.reduce_mean(-tf.log(sig_real_class))
    return [Loss(loss_g, 'generator_loss', restrict_to=[generator]),
            Loss(loss_d, 'descriminator_loss', restrict_to=[descriminator])]

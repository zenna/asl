def tb_loss(i, writer, loss, **kwargs):
  "Plot loss on tensorboard"
  writer.add_scalar('data/scalar1', loss.data[0], i)

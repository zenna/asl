class Boolean():
  pass

class BooleanEnum(Enum):
  "Boolean"
  yes = 0
  no = 1

class Integer(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 1, 1)))
  "Boolean"
  yes = 0
  no = 1

import asl
"Primitive Types"

class Integer(asl.Type):
  "Integer"
  # typesize = (nimages,)
  typesize = (3,) # HACK

# def IntegerOneHot1D(typesize):
# encode(Integer, )

class IntegerOneHot1D(Integer, asl.OneHot1D):
  typesize = (3,)
  def __init__(self, value, expand_one=True):
    self.value = asl.util.maybe_expand(IntegerOneHot1D, value, expand_one)

# class Integer():
#   pass


# encoding = asl.OneHot1D
# Integer = asl.encode(clevr.Integer, encoding, (8,))

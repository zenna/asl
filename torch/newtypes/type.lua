local util = require "util"
local constructor = util.constructor
local module = {}

-- Type
------
local Type = {}
Type.__index = Type

function Type.new(name)
  local self = setmetatable({}, Type)
  -- self.shape = shape
  self.name = name
  return self
end
constructor(Type)
module.Type = Type

local ConstrainedType = {}
ConstrainedType.__index = ConstrainedType

function ConstrainedType.new(name, shape, dtype)
  local self = setmetatable({}, ConstrainedType)
  self.name = name
  self.shape = shape
  self.dtype = dtype
  return self
end
constructor(ConstrainedType)
module.ConstrainedType = ConstrainedType

function ConstrainedType:sample(sampler, cuda_on)
  assert(self.shape ~= nil, "Need a shape to sample")
  assert(self.dtype ~= nil, "Need a dtype to sample")
  local shape = self.shape
  -- if add_batch then shape = util.add_batch(shape, batch_size) end
  local sample = sampler(shape)
  assert(sample:type() == self.dtype)
  if cuda_on then
    return sample:cuda()
  else
    return sample
  end
end

function module.constrain_types(shapes, dtypes)
  -- Convert types into constrained types
  local constrained_types = {}
  for k, v in pairs(shapes) do
    constrained_types[k] = ConstrainedType(k, shapes[k], dtypes[k])
  end
  return constrained_types
end

return module

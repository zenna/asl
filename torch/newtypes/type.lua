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

function ConstrainedType.new(name, constraints)
  local self = setmetatable({}, ConstrainedType)
  self.name = name
  self.constraints = constraints
  return self
end
constructor(ConstrainedType)
module.ConstrainedType = ConstrainedType

function ConstrainedType:sample(sampler)
  assert(self.constraints.shape ~= nil, "Need a shape to sample")
  assert(self.constraints.dtype ~= nil, "Need a dtype to sample")
  local sample = sampler(self.shape)
  assert(sample:type() == self.constraints.dtype)
  return sample
end

function module.constrain_type_shape_dtype(Type, shape, dtype)
  return ConstrainedType(Type.name, {shape=shape, dtype=dtype})
end

return module

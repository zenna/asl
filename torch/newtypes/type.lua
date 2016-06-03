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

function ConstrainedType:sample(sampler)
  assert(self.shape ~= nil, "Need a shape to sample")
  assert(self.dtype ~= nil, "Need a dtype to sample")
  local sample = sampler(self.shape)
  assert(sample:type() == self.dtype)
  return sample
end

return module

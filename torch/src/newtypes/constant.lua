local util = require "dddt.util"
local constructor = util.constructor

-- Constant
-----------
local Constant = {}
Constant.__index = Constant

function Constant.new(type, name)
  local self = setmetatable({}, Constant)
  self.type = type
  self.name = name
  return self
end
constructor(Constant)

function Constant:constrain(type_to_constrained)
  local constrained_type = type_to_constrained[self.name]
  return Constant(constrained_type. self.name)
end

return {Constant=Constant}

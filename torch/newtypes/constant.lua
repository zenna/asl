local util = require "util"
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

return Constant

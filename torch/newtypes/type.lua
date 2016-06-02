local util = require "util"
local constructor = util.constructor

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


return {Type=Type}

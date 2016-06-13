local util = require "util"
local constructor = util.constructor

-- A typed function
-------------------

local ConcreteFunc = {}
ConcreteFunc.__index = ConcreteFunc

function ConcreteFunc.new(interface, func)
  local self = setmetatable({}, ConcreteFunc)
  self.interface = interface
  self.func = func -- f: X x theta -> Y
  return self
end
constructor(ConcreteFunc)

function ConcreteFunc.fromParamFunc(param_func, params)
  -- print(param_func, params)
  local func = function(input)
    return param_func.param_func(params, input)
  end
  return ConcreteFunc(param_func.interface, func)
end

-- Can I overload call?
function ConcreteFunc:call(input)
  -- print("Calling %s" % self.interface.name)
  local g = self.func(input)
  return g
end

return {ConcreteFunc=ConcreteFunc}

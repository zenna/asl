local util = require "util"
local constructor = util.constructor
local common = require "templates.common"
local t = require "torch"

--A parameterised function (space)
----------------------------------

-- A parameterised function space represented a set of functions {f:X -> Y}
-- as a mapping f:theta -> (X -> Y).
-- We can represent function spaces as normal functions which take an extr
-- parameter input, i.e. f: X x theta -> Y

local ParamFunc = {}
ParamFunc.__index = ParamFunc

function ParamFunc.new(interface, param_func)
  local self = setmetatable({}, ParamFunc)
  self.interface = interface
  self.param_func = param_func -- f: X x theta -> Y
  return self
end
constructor(ParamFunc)

-- -- -- Generated a parameterised type from a set of types
-- function ParamFunc.fromTypes()
--   ...
-- end

-- Can I overload call?
function ParamFunc:call(input, params)
  print("Calling %s" % self.interface.name)
  return self.param_func(input, params)
end

-- Generate a set of params
function ParamFunc:gen_params()
  local inp_shapes = self.interface:inp_shapes()
  local faux_inputs = util.map(t.rand, inp_shapes)
  local params = common.gen_param()
  self.param_func(faux_inputs, params)
  return util.update({}, params)
end

return {ParamFunc=ParamFunc}

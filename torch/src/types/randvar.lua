local util = require "pdt.util"
local constructor = util.constructor

-- Random Variable
------------------
local RandVar = {}
RandVar.__index = RandVar

function RandVar.new(type, name)
  local self = setmetatable({}, RandVar)
  self.type = type
  self.name = name
  -- self.is_stale = true
  self.gen = function()
    error("No distribution assigned to %s::%s" % {self.type.name, self.name})
  end
  return self
end
constructor(RandVar)

-- A transformed randvar is one thats been modifed by an un
local TransformedRandVar = {}
TransformedRandVar.__index = TransformedRandVar

function TransformedRandVar.new(interface, inp_randvars, out_index)
  local self = setmetatable({}, TransformedRandVar)
  self.interface = interface
  self.inp_randvars = inp_randvars
  self.out_index = out_index
  self.type = self.interface.rhs[out_index]
  return self
end
constructor(TransformedRandVar)

function TransformedRandVar:gen(funcs)
  local func = funcs[self.interface.name]
  assert(func ~= nil)
  local values = util.map(function(rv) return rv:gen(funcs) end, self.inp_randvars)
  local result = func:call(values)[self.out_index]
  assert(result ~= nil)
  return result
end


function RandVar:set_params(params)
  self.params = params
end

return {RandVar=RandVar, TransformedRandVar=TransformedRandVar}

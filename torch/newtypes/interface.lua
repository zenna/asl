local util = require "util"
local constructor = util.constructor
local randvar = require("./randvar")
local ParamRandVar = randvar.ParamRandVar

-- Interface
------------
local Interface = {}
Interface.__index = Interface

-- An interface is an implicitly represented set of functions
function Interface.new(lhs, rhs, name)
  local self = setmetatable({}, Interface)
  print("Creating interface %s" % name)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  return self
end
constructor(Interface)

-- Applying an interface to a randvar yields a (set of) randvars(s)
function Interface:call(inp_randvars)
  print("Applying function %s" % self.name)
  local randvars = {}
  -- Type check
  assert(util.all_eq(util.extract('type', inp_randvars), self.lhs))

  -- For every output construct a random variable
  for i = 1, #self.rhs do
    local r = ParamRandVar(self.rhs[i], inp_randvars, i)
    table.insert(randvars, r)
  end
  return randvars
end

function Interface:inp_shapes()
  return util.map(function(x) return x.shape end, self.lhs)
end

function Interface:out_shapes()
  return util.map(function(x) return x.shape end, self.rhs)
end

function constrain_interface(interface, type_to_constrained)
  local replace = function(type)
    local constrained_type = type_to_constrained[type.name]
    assert(constrained_type ~= nil)
    return constrained_type
  end
  local lhs = util.map(replace, interface.lhs)
  local rhs = util.map(replace, interface.rhs)
  return Interface(lhs, rhs, interface.name)
end

-- Concrete Interface
---------------------

local ConcreteInterface = {}
ConcreteInterface.__index = Interface

-- An interface is an actual functions
function ConcreteInterface.new(interface, func, params)
  local self = setmetatable({}, ConcreteInterface)
  self.interface = interface
  self.func = func
  self.params = params
  return self
end
constructor(ConcreteInterface)

function ConcreteInterface:call(input)
  return self.func(input, self.params)
end

return {ConcreteInterface=ConcreteInterface, Interface=Interface}

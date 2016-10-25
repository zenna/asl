local util = require "pdt.util"
local constructor = util.constructor
local randvar = require("pdt.types.randvar")
local TransformedRandVar = randvar.TransformedRandVar

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

local function cartprodstring(types)
  return table.concat(util.extract('name', types), " × ")
end

function Interface:__tostring()
  return "%s :: %s -> %s" % {self.name, cartprodstring(self.lhs),
                             cartprodstring(self.rhs)}
end


-- Applying an interface to a randvar yields a (set of) randvars(s)
function Interface:call(inp_randvars)
  print("Applying function %s" % self.name)
  local randvars = {}
  -- Type check
  assert(util.all_eq(util.extract('type', inp_randvars), self.lhs))

  -- For every output construct a random variable
  for i = 1, #self.rhs do
    print("heres")
    local r = TransformedRandVar(self, inp_randvars, i)
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

function Interface:constrain(type_to_constrained)
  local replace = function(type)
    local constrained_type = type_to_constrained[type.name]
    assert(constrained_type ~= nil)
    return constrained_type
  end
  local lhs = util.map(replace, self.lhs)
  local rhs = util.map(replace, self.rhs)
  return Interface(lhs, rhs, self.name)
end

return {Interface=Interface}

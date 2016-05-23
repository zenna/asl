local dddt = {}
-- libraries:
local t = require "torch"
local util = "util"
-- grad = require 'autograd'


local function constructor(atype)
  setmetatable(atype, {
    __call = function (cls, ...)
      return cls.new(...)
    end,
  })
end

local Type = {} -- the table representing the class, which will double as the metatable for the instances
Type.__index = Type -- failed table lookups on the instances should fallback to the class table, to get methods

-- syntax equivalent to "MyClass.new = function..."
function Type.new(shape, name)
  local self = setmetatable({}, Type)
  self.shape = shape
  self.name = name
  return self
end
constructor(Type)
dddt.Type = Type

-- Interface
local Interface = {}
Interface.__index = Interface

function Interface.new(lhs, rhs, name, template_kwargs)
  local self = setmetatable({}, dddt.Interface)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  self.template = template_kwargs.template
  self.params = template_kwargs.gen_params()
  self.template_kwargs = template_kwargs
  return self
end
constructor(Interface)

function type_check(randvars, types)
  assert(#randvars == #types)
  for i = 1, #randvars do
    assert(randvars[i].type == types[i])
  end
end

-- function type_check(lhs, rhs)
--   print(lhs, rhs)
--   assert(#lhs == #rhs)
--   for i = 1, #lhs do
--     assert(lhs[i].type == rhs[i].type)
--   end
-- end

function types(typed_vals)
  local types = {}
  for u = 1, #typed_vals do
    table.insert(types, typed_vals[i].type)
  end
  return types
end

function Interface:call(inp_randvars)
  print("Applying function %s" % self.name)
  local randvars = {}
  type_check(inp_randvars, self.lhs)
  print("before inp_randvars", inp_randvars)
  for i = 1,#self.rhs do
    local r = dddt.RandVar(self.rhs[i])
    r.get_value = function()
      inp_randvars_vals = {}
      for j = 1, #inp_randvars do
        local q = inp_randvars[j].get_value()
        table.insert(inp_randvars_vals, q)
      end
      print("adada", inp_randvars_vals)
      return self.template(inp_randvars_vals, self.params)
    end
    table.insert(randvars, r)
  end
  return randvars
end
dddt.Interface = Interface

-- Abstract Data Type
local AbstractDataType = {}
AbstractDataType.__index = AbstractDataType

function AbstractDataType.new(funcs, consts, randvars, axioms, name)
  local self = setmetatable({}, AbstractDataType)
  self.funcs = funcs
  self.consts = consts
  self.randvars = randvars
  self.axioms = axioms
  self.name = name
  return self
end
constructor(AbstractDataType)
dddt.AbstractDataType = AbstractDataType

local RandVar = {}
RandVar.__index = RandVar

function RandVar.new(type, name)
  local self = setmetatable({}, RandVar)
  self.type = type
  self.name = name
  self.get_value = function()
    error("No distribution assigned to $s::$s" % {self.type.name, self.name})
  end
  return self
end
constructor(RandVar)
dddt.RandVar = RandVar

local function distribution_rand_var(type, sampler)
  local rv = RandVar(type)
  rv.get_value = sampler
end

local function constant_rand_var(type, constant)
  local rv = RandVar(type)
  rv.get_value = function() return constant end
  return rv
end

-- axioms
local Axiom = {}
Axiom.__index = Axiom
function Axiom.new(lhs, rhs, name)
  local self = setmetatable({}, Axiom)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  assert(#lhs == #rhs)
  return self
end
constructor(Axiom)
dddt.Axiom = Axiom

function Axiom:naxioms()
  return #self.lhs
end

function dddt.get_losses(axiom, dist)
  -- returns a function that can be applied to radnom variable values
  print("axiom", axiom)
  local losses = {}
  for i = 1, axiom:naxioms() do
    print("EYE",  i)
    print(axiom.lhs[i].get_value())
    print(axiom.rhs[i].get_value())
    table.insert(losses, dist(axiom.lhs[i].get_value(),
                              axiom.rhs[i].get_value()))
  end
  return losses
end

return dddt

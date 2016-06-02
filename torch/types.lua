local dddt = {}
-- libraries:
local t = require "torch"
local util = require "util"
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}
local constructor = util.constructor



-- We need a simple implementation of a concrete data type and a separation
-- of concerns
-- An interface should specify: the types of the input and output
-- There's an assignment of a function space
-- And then there's an assignment of a function
-- Currently it also specifies a function space

-- Types
-- Type: A data type.  Pretty much just a name
-- Interface: a functional type specification for a name
-- Constant: a functional type specification for a constant
-- ADT = ({T}, {O}, {F})

-- Should a type contain shape information?

-- Then we need something which assigns a function space to an face
-- and a value space to constants

-- Then we need concrete implementation

-- Type
------
local Type = {}
Type.__index = Type

function Type.new(shape, name)
  local self = setmetatable({}, Type)
  self.shape = shape
  self.name = name
  return self
end
constructor(Type)
dddt.Type = Type

function Type:get_shape(add_batch, batch_size)
  if add_batch then
    util.add_batch(self.shape, batch_size)
  else
    return self.shape
  end
end

local function types(typed_vals)
  local types = {}
  for u = 1, #typed_vals do
    table.insert(types, typed_vals[i].type)
  end
  return types
end

local function type_check(randvars, types)
  assert(#randvars == #types)
  for i = 1, #randvars do
    assert(randvars[i].type == types[i])
  end
end

-- Interface
------------
local Interface = {}
Interface.__index = Interface

function Interface.new(lhs, rhs, name, template_kwargs)
  local self = setmetatable({}, dddt.Interface)
  print("Creating interface %s" % name)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  -- Add shapes to templates for function/param generators
  local inp_shapes = util.map(function (x) return x:get_shape(false) end, self.lhs)
  local out_shapes = util.map(function (x) return x:get_shape(false) end, self.rhs)
  local update_template_kwargs = util.update(template_kwargs,
    {inp_shapes = inp_shapes, out_shapes = out_shapes})
  local template, params = template_kwargs.template_gen(update_template_kwargs)
  self.template = template
  self.template_kwargs = template_kwargs
  return self
end
constructor(Interface)


function Interface:call(inp_randvars)
  -- Applying an interface function a random variable yields a (set of)
  -- random variables
  print("Applying function %s" % self.name)
  local randvars = {}
  type_check(inp_randvars, self.lhs)

  -- For every output construct a random variable
  -- A random variable has a gen function which draws a value from its
  -- distribution.
  -- The output of a call will be a "functional" random variable.
  -- That is, if X is a random variable, this will be f(X).
  -- However if f is a parameterised function, we need these parameters
  -- to generate f(x)
  -- Option 1. Let gen take parameter values.  Problem not all generators
  --   need paramaters
  for i = 1, #self.rhs do
    local r = dddt.RandVar(self.rhs[i])
    r.gen = function()
      -- Check whether any of my inputs have changed (if not I needn't change)
      local inps_changed = false
      for j = 1, #inp_randvars do
        inps_changed = inps_changed or inp_randvars[j].is_stale
      end
      -- Only recompute if I'm not initialised (is_stale) or inputs have changed
      if inps_changed or r:is_stale() then
        local inp_randvars_vals = {}
        for j = 1, #inp_randvars do
          local q = inp_randvars[j].gen()
          table.insert(inp_randvars_vals, q)
        end
        local val = self.template(inp_randvars_vals, r:params())[i]
        r:set_value(val)
        print("regenerating")
        return val
      else
        print("Using Cache")
        return r:value()
      end
    end
    table.insert(randvars, r)
  end
  return randvars
end

function Interface:get_params()
  return self.params
end
dddt.Interface = Interface

-- Abstract Data Type
---------------------
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

function AbstractDataType:get_params()
  local params = {}
  for i = 1, #self.funcs do
    local func = self.funcs[i]
    assert(params[func.name] == nil)
    params[func.name] = func:get_params()
  end
  for i = 1, #self.consts do
    local const = self.consts[i]
    assert(params[const.name] == nil)
    params[const.name] = const:get_params()
  end
  return params
end

-- Random Variable
------------------
local RandVar = {}
RandVar.__index = RandVar

function RandVar.new(type, name)
  local self = setmetatable({}, RandVar)
  self.type = type
  self.name = name
  self.is_stale = true
  self.gen = function()
    error("No distribution assigned to $s::$s" % {self.type.name, self.name})
  end
  return self
end
constructor(RandVar)
dddt.RandVar = RandVar

function RandVar:value()
  if self.is_stable then
    error("Pulling stale value")
  else
    return self.val
  end
end

function RandVar:set_value(val)
  self.val = val
  self.is_stable = false
end

function RandVar:set_params(params)
  self.params = params
end

function RandVar:params()
  return self.params
end

-- Axioms
---------
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
    print("evaluating axiom: %s" %i)
    local lhs_val = axiom.lhs[i].gen()
    local rhs_val  = axiom.rhs[i].gen()
    local loss = dist(lhs_val, rhs_val)
    print("loss", loss)
    table.insert(losses, loss)
  end
  return losses
end

function dddt.get_loss_fn(axioms, dist)
  print("dist", dist)
  -- Returns a loss function loss(params)
  local loss_fn = function(params)
    print("params here", params)
    local losses = {}
    for i = 1, #axioms do
      local loss = dddt.get_losses(axioms[i], dist)
      table.insert(losses, loss)
    end
    return t.Tensor(losses):sum()
  end
  return loss_fn
end



return dddt

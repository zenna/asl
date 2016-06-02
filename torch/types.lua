local dddt = {}
-- libraries:
local t = require "torch"
local util = require "util"
local constructor = util.constructor

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
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  -- Add shapes to templates for function/param generators
  local inp_shapes = util.map(function (x) return x:get_shape(false) end, self.lhs)
  local out_shapes = util.map(function (x) return x:get_shape(false) end, self.rhs)
  local update_template_kwargs = util.update(template_kwargs,
    {inp_shapes = inp_shapes, out_shapes = out_shapes})
  template, params = template_kwargs.template_gen(update_template_kwargs)
  self.template = template
  self.params = params
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
        local val = self.template(inp_randvars_vals, self.params)[i]
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
    local losses = {}
    for i = 1, #axioms do
      local loss = dddt.get_losses(axioms[i], dist)
      table.insert(losses, loss)
    end
    return t.Tensor(losses):sum()
  end
  return loss_fn
end

-- Param
local function parsename(x)
  -- Expects parameter name in form name_1,2,3,
  local splitted = util.split(x,"_")
  assert(#splitted == 2)
  local id = splitted[1]
  local shape_str = util.split(splitted[2], ",")
  local shape = util.map(tonumber, shape_str)
  return id, shape
end


function default_index(tbl, k)
  local id, shape = parsename(k)
  local new_val = t.rand(t.LongStorage(shape))
  tbl[k] = new_val
  return new_val
end

function dddt.gen_param()
  local param = {}
  setmetatable(param,{
  __index = function(param,k) return default_index(param, k) end
  })
  return param
end

--
-- local Param = {}
-- Param.__index = Param
-- function Param.new()
--   local self = setmetatable({}, Param)
--   return self
-- end
-- -- constructor(Param)
-- setmetatable(Param,{
--   __call = function (cls, ...)
--     return cls.new(...)
--   end,
--   __index = function(t,k) return 0 end,
--   -- __index = default_index
-- })
-- dddt.Param = Param


return dddt

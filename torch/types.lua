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
function Type.new(shape, dtype, name)
  local self = setmetatable({}, Type)
  self.shape = shape
  self.dtype = dtype
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

function Interface:call(randvar)
  local f = function(x)
    return self.template(x, self.params)
  end
  return RandVar( )
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

function RandVar.new(type)
  local self = setmetatable({}, RandVar)
  self.type = type
  self.call = util.identity
  return self
end
constructor(RandVar)
dddt.RandVar = RandVar

local function constant_rand_var(type, constant)
  local rv = RandVar(type)
  rv.get_value = function() return constant end
  return rv
end

return dddt

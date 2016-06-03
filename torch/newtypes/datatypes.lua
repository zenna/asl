local util = require "util"
local constructor = util.constructor

-- Abstract Data Type
---------------------
local AbstractDataType = {}
AbstractDataType.__index = AbstractDataType

function AbstractDataType.new(interfaces, constants, name)
  local self = setmetatable({}, AbstractDataType)
  self.interfaces = interfaces
  self.constants = constants
  self.name = name
  -- All typenames should be unique
  local interface_names = util.map(function(x) return x.name end, interfaces)
  local constant_names = util.map(function(x) return x.name end, constants)
  assert(util.all_unique(interface_names))
  assert(util.all_unique(constant_names))
  return self
end
constructor(AbstractDataType)

function AbstractDataType:types()
  -- Get typenames
  assert(false)
end

-- Concrete Data Type
---------------------
local ConcreteDataType = {}
ConcreteDataType.__index = ConcreteDataType

function ConcreteDataType.new(adt, funcs, constants)
  local self = setmetatable({}, ConcreteDataType)
  assert(#funcs == #adt.interfaces)
  assert(#constants == #adt.constants)
  self.funcs = funcs
  self.consts = constants
  return self
end
constructor(ConcreteDataType)

-- Construct a concrete data type from a param whose keys are the names
function ConcreteDataType.from_params(adt, params)
  -- params = {push}
  assert(false)
end

return {DataType=DataType, AbstractDataType=AbstractDataType,
        ConcreteDataType=ConcreteDataType}

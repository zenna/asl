local util = require "util"
local constructor = util.constructor

-- Abstract Data Type
---------------------
local AbstractDataType = {}
AbstractDataType.__index = AbstractDataType

function AbstractDataType.new(typenames, interfaces, constants, name)
  local self = setmetatable({}, AbstractDataType)
  self.typenames = typenames
  self.interfaces = interfaces
  self.constants = constants
  self.name = name
  -- All typenames should be unique
  local typename_names = util.map(function(x) return x.name end, typenames)
  local interface_names = util.map(function(x) return x.name end, interfaces)
  local constant_names = util.map(function(x) return x.name end, constants)
  assert(util.all_unique(typename_names))
  assert(util.all_unique(interface_names))
  assert(util.all_unique(constant_names))
  return self
end
constructor(AbstractDataType)

-- Data Type
---------------------
local DataType = {}
DataType.__index = DataType

-- A DataType assigns shapes to the types. Stack : Shape = {1, 10}
-- But for a function like push, it seems less like a data type
-- In the sense that yeah I say say {1,2} is an element of type Stack
-- But can I assign a function and say it is of type push
function DataType.new(adt, shapes)
  local self = setmetatable({}, DataType)
  self.adt = adt
  self.shapes = shapes
  return self
end
constructor(DataType)

-- Concrete Data Type
---------------------
local ConcreteDataType = {}
ConcreteDataType.__index = ConcreteDataType

function ConcreteDataType.new(adt, funcs, constants)
  local self = setmetatable({}, ConcreteDataType)
  assert(#funcs == #adt.interfaces)
  assert(#constants == #adt.constants)
  self.funcs = funcs
  self.consts = consts
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

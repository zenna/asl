local util = require "util"
local constructor = util.constructor

-- Abstract Data Type
---------------------
local AbstractDataType = {}
AbstractDataType.__index = AbstractDataType

local function all_unique(tbl)
  for i, v in ipairs(tbl) do
    assert(false)
  end
end

function AbstractDataType.new(typenames, interfaces, constants, name)
  local self = setmetatable({}, AbstractDataType)
  self.typenames = typenames
  self.interfaces = interfaces
  self.constants = constants
  self.name = name
  -- All typenames should be unique
  all_unique(typenames)
  assert(#interfaces)
  return self
end
constructor(AbstractDataType)

-- Data Type
---------------------
local DataType = {}
DataType.__index = DataType

local function all_unique(tbl)
  for i, v in ipairs(tbl) do
    assert(false)
  end
end

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
  assert(#constants == $adt.constants)
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

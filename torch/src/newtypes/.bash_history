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

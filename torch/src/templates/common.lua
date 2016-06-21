local t = require "torch"
local util = require "dddt.util"
local common = {}
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

local function default_index(tbl, k)
  print("generating paramter values")
  local id, shape = parsename(k)
  local new_val = t.rand(t.LongStorage(shape)) * 0.1
  tbl[k] = new_val
  return new_val
end

function common.gen_param()
  local param = {}
  setmetatable(param,{
    __index = function(param,k) return default_index(param, k) end
  })
  return param
end

function common.param_str(id, shape)
  return "%s_%s" % {id, util.tostring(shape)}
end



local function axes_f(f, tensor, axes)
  local out = tensor
  for i, axis in ipairs(axes) do
    out = f(out, axis)
  end
  return out
end

-- deterministic batch normalisation
local function batch_norm4d(input_shape)
  local eps = 1e-4
  local axes = {1, 3, 4}
  local shape = util.shape({1, input_shape[2], 1, 1})
  local init_params = {gamma = t.ones(shape), beta = t.zeros(shape)}
  local f = function(params, input)
    local mean = axes_f(t.mean, input, axes):expandAs(input)
    local var = axes_f(t.var, input, axes):expandAs(input)
    local gamma = params.gamma:expandAs(input)
    local beta = params.beta:expandAs(input)
    local inv_std = torch.cinv(torch.sqrt(var + eps))
    return t.cmul(input - mean, t.cmul(gamma, inv_std) + beta)
  end
  return f, init_params
end

img = torch.rand(512,3,28,28)
f, p = batch_norm4d(img:size())
print("testbn", f(p, img):size())

return common

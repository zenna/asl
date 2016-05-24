local t = require "torch"
local util = {}

function util.constructor(atype)
  setmetatable(atype, {
    __call = function (cls, ...)
      return cls.new(...)
    end,
  })
end

function util.shape(shape_tbl)
  return t.LongStorage(shape_tbl)
end

function util.identity(x)
  return x
end

function util.circular_indices(lb, ub, thresh)
  local indices
  local curr_lb = lb
  local stop
  local i = 1
  while true do
    stop = math.min(ub, thresh)
    print(curr_lb, stop, ub)
    local ix = t.range(curr_lb, stop):long()
    if i == 1 then -- ew
      indices = ix
    else
      indices = t.cat(indices, ix)
    end
    i = i + 1
    if stop ~= ub then
      local diff = ub - stop
      curr_lb = lb
      ub = diff + lb - 1
    else
      break
    end
  end
  return indices, stop
end

function util.add_batch(shape, batchsize)
  local new_shp = t.LongStorage(#shape + 1)
  new_shp[1] = batchsize
  for i = 2, #new_shp do
    new_shp[i] = shape[i-1]
  end
  return new_shp
end

return util

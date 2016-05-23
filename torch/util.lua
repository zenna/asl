local t = require "torch"
local dddt = {}

function dddt.shape(shape_tbl)
  return t.LongStorage(shape_tbl)
end

function dddt.identity(x)
  return x
end

return dddt

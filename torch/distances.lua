dbg = require "debugger"
local t = torch
distances = {}
function distances.mse(a, b)
  local a_b = a - b
  z = t.mean(t.abs(a_b))
  -- dbg()
  return z
end

return distances

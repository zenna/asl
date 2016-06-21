local t = require("torch")
local distances = {}
function distances.mae(a, b)
  local a_b = a - b
  local z = t.mean(t.abs(a_b))
  -- dbg()
  return z
end

function distances.mse(a, b)
  local eps = 1e-9
  local a_b = a - b
  local z = t.mean(t.cmax(t.cmul(a_b, a_b), eps))
  -- dbg()
  return z
end

return distances

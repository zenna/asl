local t = require("torch")
local distances = {}
function distances.mae(a, b)
  local a_b = a - b
  local z = t.mean(t.abs(a_b))
  -- dbg()
  return z
end

function distances.mse(a, b)
  local a_b = a - b
  local z = t.mean(t.cmul(a_b, a_b))
  -- dbg()
  return z
end

return distances

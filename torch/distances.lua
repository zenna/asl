local t = torch
distances = {}
function distances.mse(a, b)
  local a_b = a - b
  return t.mean(t.cmul(a_b, a_b))
end

return distances

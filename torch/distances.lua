function mse(a, b)
  print(a:size(),b:size())
  local a_b = a - b
  return t.cmul(a_b, a_b):sum()
end

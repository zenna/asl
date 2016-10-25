pdt = require "types"

p = pdt.gen_param()
p[1] = 3
print(p)
x = "thisisthename1-2-2_1,2,3,4"
b = p[x]
print(p)

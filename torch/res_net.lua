local dddt = require "types"
local t = require "torch"
dddt.res_net = {}

function dddt.res_net.nnet(x, params)
  local h1 = t.tanh(x * params.W[1] + params.b[1])
  local h2 = t.tanh(h1 * params.W[2] + params.b[2])
  local yHat = h2 - t.log(t.sum(t.exp(h2)))
  return yHat
end

function dddt.res_net.net_params()
  local params = {
      W = {
        t.randn(100,50),
        t.randn(50,10),
      },
      b = {
        t.randn(50),
        t.randn(10),
      }
    }
  return params
end

return dddt.res_net

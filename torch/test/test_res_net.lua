local util = require "util"
local map = util.map
local res_net = require "res_net"
local grad = require "autograd"
local distances = require "distances"
local t = require "torch"
local templates = require "templates"

local function test_rest_net()
  local s1 = util.shape({1, 5, 5})
  local s2 = util.shape({1, 5, 5})
  local s3 = util.shape({1,5,5})
  local inp_shapes = {s1, s2}
  local o1 = util.shape({1, 50, 50})
  local o2 = util.shape({1,5,5})
  local out_shapes = {o2}
  local kwargs = {}
  kwargs['inp_shapes'] = inp_shapes
  kwargs['out_shapes'] = out_shapes
  kwargs['layer_width'] = 10
  kwargs['block_size'] = 2
  kwargs['nblocks'] = 2

  local func = res_net.gen_res_net(kwargs)
  -- local batch_inp_shapes = map(function(x) return util.add_batch(x,1) end,
  --                              inp_shapes)
  -- local inputs = map(t.rand, batch_inp_shapes)
  -- local params = templates.gen_param()
  -- local result = func(inputs, params)
  -- local loss_fn = function(params, inputs)
  --   return t.sum(func(inputs, params)[1])
  -- end
  -- local loss_fn_grad = grad(loss_fn)
  -- print(loss_fn_grad(params, inputs))
end

test_rest_net()

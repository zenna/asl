-- This is a template for a resial multi layer neural network
-- It is parameterised by the number of blocks, layers per block,
-- and inner layeer widths

-- Ways to do this.  We make a function generator which takes hyper parameters
-- and returns a function
local res_net = {}
local t = require("torch")
local util = require("util")
local templates = require("templates")
local map = util.map

local function batch_flatten(x)
  -- Converts a tensor of size (batchsize, d2, d3, d4, ...) to
  -- (batchsize, d3 * d3 * d4 * ...)
  assert(x:nDimension() > 1)
  local batch_size = x:size()[1]
  return x:view(batch_size, -1)
end

local function concatenate_inputs(inputs)
  local flat_inputs = map(batch_flatten, inputs)
  local flat_input = t.cat(flat_inputs, 2)
  return flat_input
end

local function dense_layer(input, ninputs, noutputs, params, sfx)
  -- print("inp", input:size())
  -- print("weight", params[param_str('W-%s' % sfx, {ninputs, noutputs})]:size())
  local matmul = input * params[param_str('W-%s' % sfx, {ninputs, noutputs})]
  -- print("mm", matmul:size())
  -- print("bias", params[param_str('b-%s' % sfx, {1, noutputs})]:size())
  local bias = matmul + params[param_str('b-%s' % sfx, {1, noutputs})]:expandAs(matmul)
  return t.sigmoid(bias)
end

-- Want:
-- to produce a normal function in the end that just takes in a table of params
-- to be able to generate the parameters From the function
-- Somewhere we need to specfiy the initialisation

local function flat_shapes(shape)
  return map(function(v)
     return t.prod(t.LongTensor(v)[{{2,-1}}]) end, shape)
end

function res_net.gen_res_net(kwargs)
  -- A residual network of n inputs and moutputs
  local inp_shapes = kwargs['inp_shapes']
  local out_shapes = kwargs['out_shapes']
  local layer_width = kwargs['layer_width']
  local nblocks = kwargs['nblocks']
  local block_size = kwargs['block_size']
  local ninputs = #inp_shapes
  local noutputs = #out_shapes

  local flat_input_shapes = flat_shapes(inp_shapes)
  local input_width = t.sum(t.LongTensor(flat_input_shapes))
  local flat_output_shapes = flat_shapes(out_shapes)
  local output_width = t.sum(t.LongTensor(flat_output_shapes))

  local res_net_func = function(inputs, params)
    -- Flatten and concatenate inputs
    local flat_input = concatenate_inputs(inputs)
    local data_input_width = flat_input:size()[2]
    assert(data_input_width == input_width)
    assert(#inputs == ninputs)

    -- Project input into inner layer widths
    local prev_layer = flat_input
    local wx
    if layer_width ~= input_width then
      wx = dense_layer(flat_input, input_width, layer_width, params, 'wxinpproj')
      prev_layer = dense_layer(flat_input, input_width, layer_width, params, 'inpproj')
    else
      wx = prev_layer
    end

    -- Inner Layers
    for i = 1, nblocks do
      for j = 1, block_size do
        local sfx = '%s-%s' % {j, i}
        prev_layer = dense_layer(prev_layer, layer_width, layer_width, params, sfx)
      end
      prev_layer = wx + prev_layer
    end

    -- Output Projection
    local output_product
    if layer_width ~= output_width then
      local wx_sfx = 'wxoutproj'
      output_product = dense_layer(prev_layer, layer_width, output_width, params, wx_sfx)
    else
      output_product = prev_layer
    end

    -- Output Slicing
    local outputs = {}
    local lb = 1
    for i = 1, noutputs do
      -- FIXME, make this work with views and no copying
      local ub = lb + flat_output_shapes[i] - 1
      local out = output_product[{{},{lb,ub}}]
      local rout = t.reshape(out, out_shapes[i])
      table.insert(outputs, rout)
      lb = ub + 1
    end

    return outputs
  end
  return res_net_func
end

local function test_rest_net()
  local s1 = util.shape({5,10,23})
  local s2 = util.shape({5,100})
  local s3 = util.shape({5,5,5,5})
  local inp_shapes = {s1, s2, s3}
  local o1 = util.shape({5,6})
  local o2 = util.shape({5,10,3})
  local out_shapes = {o1, o2}
  local kwargs = {}
  kwargs['inp_shapes'] = inp_shapes
  kwargs['out_shapes'] = out_shapes
  kwargs['layer_width'] = 10
  kwargs['block_size'] = 2
  kwargs['nblocks'] = 2

  local func = res_net.gen_res_net(kwargs)
  local inputs = map(t.rand, inp_shapes)
  local params = templates.gen_param()
  local result = func(inputs, params)
  print(result)
end

test_rest_net()

return res_net

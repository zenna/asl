-- Residual Convolutional Neural Network Template
-------------------------------------------------

-- This is a template for a resial multi layer neural network
local conv_res_net = {}
local autograd = require("autograd")
local nn = autograd.nn
local model = autograd.model
if not cutorch then
   require 'cutorch'
   runtests = true
end
local util = require("util")
local common = require("./common")
local param_str = common.param_str
local map = util.map

local function isimgbatch(input)
  return input:nDimension() == 4
end


-- A residual network of n inputs and moutputs
function conv_res_net.gen_conv_res_net(interface, kwargs)
  local inp_shapes = interface:inp_shapes()
  local out_shapes = interface:out_shapes()
  local ninputs = #inp_shapes
  local noutputs = #out_shapes

  -- Parameters
  local args = util.update({inputFeatures=ninputs}, kwargs)
  -- local nblocks = kwargs['nblocks']
  -- local block_size = kwargs['block_size']
  -- local width, height = kwargs['width'], kwargs['height']
  local cnet, params = model.SpatialNetwork(args)


  local res_net_func = function(params, inputs)
    -- print("Calling ConvNet")
    -- check input are good
    -- dbg()
    assert(util.all(map(isimgbatch, inputs)))

    -- Get inputs
    -- dbg()
    local inp_img = torch.cat(inputs, 2)

    local output_product = cnet(params, inp_img)

    local outputs = {}
    for i = 1,noutputs do
      local output_slice = output_product[{{},{i,i},{},{}}]
      table.insert(outputs, output_slice)
    end
    -- dbg()
    return outputs
  end
  return res_net_func, params
end

return conv_res_net

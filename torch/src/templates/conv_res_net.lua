-- Residual Convolutional Neural Network Template
-------------------------------------------------

-- This is a template for a resial multi layer neural network
local conv_res_net = {}
local util = require("dddt.util")
local autograd = require("autograd")
local common = require("dddt.templates.common")
local model = autograd.model

local function isimgbatch(input)
  return input:nDimension() == 4
end

local function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- A residual network of n inputs and moutputs
function conv_res_net.gen_conv_res_net(interface, kwargs)
  local inp_shapes = interface:inp_shapes()
  local out_shapes = interface:out_shapes()
  local ninputs = #inp_shapes
  local noutputs = #out_shapes

  -- Parameters
  local args = util.update({inputFeatures=ninputs}, kwargs)
  local hf = args['hiddenFeatures']
  if hf == nil then
    args['hiddenFeatures'] = {noutputs}
  else
    local a = shallowcopy(hf)
    table.insert(a, noutputs)
    args['hiddenFeatures'] = a
  end

  -- local nblocks = kwargs['nblocks']
  -- local block_size = kwargs['block_size']
  -- local width, height = kwargs['width'], kwargs['height']
  local cnet, params = model.SpatialNetwork(args)


  local res_net_func = function(params, inputs)
    -- print("Calling ConvNet")
    -- check input are good
    -- dbg()
    assert(util.all(util.map(isimgbatch, inputs)))

    -- Get inputs
    -- dbg()
    local inp_img = torch.cat(inputs, 2)

    local output_product = cnet(params, inp_img)
    -- print("PRODSIZ", output_product:size())
    local outputs = {}
    for i = 1,noutputs do
      local output_slice = output_product[{{},{i,i},{},{}}]
      -- print("SLICY", output_slice.value:size())
      -- print("value", torch.sum(output_slice).value)
      table.insert(outputs, output_slice)
    end
    -- dbg()
    return outputs
  end
  return res_net_func, params
end

return conv_res_net

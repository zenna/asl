local t = require "torch"
local dddt = require "dddt"
local util = dddt.util
local Type = dddt.types.Type
local eq_axiom = dddt.types.eq_axiom
local Interface = dddt.types.Interface
local RandVar = dddt.types.RandVar
local AbstractDataType = dddt.types.AbstractDataType
local Spec = dddt.types.Spec
local Constant = dddt.types.Constant
local constrain_types = dddt.types.constrain_types
local gen_param_funcs = dddt.types.gen_param_funcs
local train = dddt.train

require "cunn"
if not cutorch then
   require 'cutorch'
   runtests = true
end

-- Genereates the stack abstract data type
local function stack_adt()
  local Stack = Type('Stack')
  local Item = Type('Item')
  local push = Interface({Stack, Item}, {Stack}, 'push')
  local pop = Interface({Stack}, {Stack, Item}, 'pop')
  local empty_stack = Constant(Stack, 'empty_stack')
  return AbstractDataType({push, pop}, {empty_stack=empty_stack}, "Stack")
end

-- Genereates the stack specification
local function stack_spec()
  local stack1 = RandVar(Type('Stack'), 'stack1')
  local item1 = RandVar(Type('Item'), 'item1')

  -- Extensional axiom
  local function axiom(funcs, randvars, constants)
    local push, pop = funcs['push'], funcs['pop']
    local items, nitems = randvars['items'], randvars['nitems']
    local stack = constants['empty_stack']
    local axioms = {}
    local pop_stack
    for i = 1, nitems do
      stack = push:call({stack, items[i]})[1]
      print("STACKSUM", torch.sum(stack).value)
      pop_stack = stack
      for j = i, 1, -1 do
        pop_stack, pop_item = unpack(pop:call({pop_stack}))
        local axiom = eq_axiom({pop_item}, {items[j]})
        print("i:%s, j:%s: loss: %s" % {i, j, axiom.value})
        table.insert(axioms, axiom)
      end
    end
    return axioms
  end
  return Spec({stack1, item1}, axiom)
end

-- Example
local function stack(shapes, dtypes, templates, template_args)
  local adt = stack_adt()
  local spec = stack_spec(adt)
  local constrained_types = constrain_types(shapes, dtypes)
  local param_funcs, interface_params = gen_param_funcs(adt.interfaces, constrained_types, templates, template_args)
  return adt, spec, constrained_types, param_funcs, interface_params
end

-- Generators
local function gen_gen(batch_size, cuda)
  local trainData = require('./get_mnist.lua')()
  local item_coroutine = dddt.generators.infinite_minibatches(trainData.x:double(), batch_size,  true)
  return function()
    local nitems = 3
    local items = {}
    for i = 1, nitems do
      local coroutine_ok, value = coroutine.resume(item_coroutine)
      if cuda then
        value = value:cuda()
      end
      assert(coroutine_ok)
      -- dbg()
      table.insert(items, value)
    end
    return {nitems=nitems, items=items}
  end
end

local function main()
  local res_net = dddt.templates.res_net
  local shapes = {Item=util.shape({1, 32, 32}), Stack=util.shape({1, 50, 50})}
  local dtypes = {Item=t.getdefaulttensortype(), Stack=t.getdefaulttensortype()}
  local batch_size = 2
  local templates = {push=res_net.gen_res_net, pop=res_net.gen_res_net}
  local template_kwargs = {}
  template_kwargs['layer_width'] = 10
  template_kwargs['block_size'] = 2
  template_kwargs['nblocks'] = 1
  local template_args = {push=template_kwargs, pop=template_kwargs}
  local adt, spec, constrained_types, param_funcs, interface_params = stack(shapes, dtypes, templates, template_args)

  -- Generator
  local generator = gen_gen(batch_size)

  -- Constants: Generate constant params
  local constant_params = {empty_stack=constrained_types['Stack']:sample(t.rand)}

  -- Generate interface params
  local interface_params = util.map(function(pf) return pf:gen_params() end, param_funcs)
  local all_params = util.update(constant_params, interface_params)
  train(param_funcs, spec.axiom, all_params, adt.constants, generator, batch_size, 10000)
end

local function conv_main()
  local conv_res_net = dddt.templates.conv_res_net
  local shapes = {Item=util.shape({1, 32, 32}), Stack=util.shape({1, 32, 32})}
  local dtypes = {Item=t.getdefaulttensortype(), Stack=t.getdefaulttensortype()}
  local batch_size = 512
  local templates = {push=conv_res_net.gen_conv_res_net,
                     pop=conv_res_net.gen_conv_res_net}
  local template_kwargs = {}
  local cuda_on = true
  template_kwargs['layer_width'] = 10
  template_kwargs['block_size'] = 2
  template_kwargs['activation'] = 'ReLU'
  template_kwargs['kernelSize'] = 3
  template_kwargs['pooling'] = 0
  template_kwargs['batchNormalization'] = true
  template_kwargs['cuda'] = cuda_on
  template_kwargs['hiddenFeatures'] = {24, 24}

  local template_args = {push=template_kwargs, pop=template_kwargs}
  local adt, spec, constrained_types, param_funcs, interface_params = stack(shapes, dtypes, templates, template_args)

  -- Generator
  local generator = gen_gen(batch_size ,cuda_on )

  -- Constants: Generate constant params
  local constant_params = {empty_stack=constrained_types['Stack']:sample(t.rand, cuda_on)}

  -- Generate interface params
  local all_params = util.update(constant_params, interface_params)
  -- dbg()
  train.train(param_funcs, spec.axiom, all_params, adt.constants, generator, batch_size, 100000)
end

conv_main()

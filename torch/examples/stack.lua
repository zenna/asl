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
local grad = require "autograd"

-- Genereates the stack abstract data type
local function stack_adt()
  local Stack = Type('Stack')
  local Item = Type('Item')
  local push = Interface({Stack, Item}, {Stack}, 'push')
  local pop = Interface({Stack}, {Stack, Item}, 'pop')
  local empty_stack = Constant(Stack, 'empty_stack')
  return AbstractDataType({push, pop}, {empty_stack=empty_stack}, "Stack")
end

local function value(x)
  if type(x) == 'number' then
    return x
  elseif type(x) == 'table' then
    return x.value
  elseif type(x) == 'userdata' then
    return x
  else
    assert(false)
  end
end

local autograd = require("autograd")
local mse = autograd.nn.MSECriterion()


-- Genereates the stack specification
local function stack_spec()
  local stack1 = RandVar(Type('Stack'), 'stack1')
  local item1 = RandVar(Type('Item'), 'item1')

  -- local function axiom(funcs, randvars, constants)
  --   local push, pop = funcs['push'], funcs['pop']
  --   local items, nitems = randvars['items'], randvars['nitems']
  --   local stack = constants['empty_stack']
  --   -- dbg()
  --   -- print("EMPTYSTACKSUM", torch.sum(stack).value)
  --   local axioms = {}
  --   -- nitems = 2
  --   for i = 1, nitems do
  --     stack = push:call({stack, items[i]})[1]
  --   end
  --   for j = nitems, 1, -1 do
  --     stack, pop_item = unpack(pop:call({stack}))
  --     local axiom = mse(pop_item, items[j])
  --     -- local axiom = torch.mean(pop_item) - torch.mean(items[j])
  --     -- axiom = axiom * axiom
  --     -- table.insert(axioms, axiom)
  --     -- axiom = eq_axiom({pop_item}, {items[j]})
  --     table.insert(axioms, axiom)
  --     print("SUMS: lhs:%s, rhs:%s" % {value(pop_item):sum(), value(items[j]):sum()})
  --     print("%sth popped: loss %s" % {j, value(axiom)})
  --   end
  -- end

  -- Extensional axiom
  local function axiom(funcs, randvars, constants)
    local push, pop = funcs['push'], funcs['pop']
    local items, nitems = randvars['items'], randvars['nitems']
    local stack = constants['empty_stack']
    -- dbg()
    -- print("EMPTYSTACKSUM", torch.sum(stack).value)
    local axioms = {}
    local pop_item
    local pop_stack
    --
    for i = 1, nitems do
      -- dbg()
      stack = push:call({stack, items[i]})[1]
      -- print("STACKSUM", torch.sum(stack).value)
      pop_stack = stack
      for j = i, 1, -1 do
        pop_stack, pop_item = unpack(pop:call({pop_stack}))
        local axiom = torch.sum(pop_item) - torch.sum(items[j])
        axiom = axiom * axiom
        -- local axiom = eq_axiom({pop_item}, {items[j]})
        local k = i - j + 1
        print("SUMS: lhs:%s, rhs:%s" % {value(pop_item):sum(), value(items[j]):sum()})
        print("%sth popped = %sth added, loss: %s" % {k, j, value(axiom)})
        table.insert(axioms, axiom)
      end
    end
    -- dbg()
    return axioms
    -- return {axioms[1]}
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
local function gen_gen(batch_size, cuda, nitems, data)
  local item_coroutine = dddt.generators.infinite_minibatches(data, batch_size,  true)
  return function()
    local items = {}
    for i = 1, nitems do
      local coroutine_ok, value = coroutine.resume(item_coroutine)
      if cuda then
        value = value:cuda()
      end
      -- print("item" .. i, torch.sum(value))
      assert(coroutine_ok)
      -- dbg()
      table.insert(items, value)
    end
    return {nitems=nitems, items=items}
  end
end

local function main()
  local cmd = t.CmdLine()
  cmd:text()
  cmd:text('Construct a stack abstract data type')
  cmd:text()
  cmd:text('Options')
  cmd:option('-optim_alg',grad.optim.adam,'Optimization algorithm')
  cmd:option('-learning_rate',0.01,'Learning Rate')
  cmd:option('-batch_size',128,'Size of minibatches')
  cmd:option('-cuda',true,'Use cuda?')
  cmd:option('-num_epochs',1000,'Number of training iterations')
  cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')

  -- Stack specific
  cmd:option('-nitems',2,'Number of items to train stack on')

  local opt = cmd:parse(arg)
  local cmd = t.CmdLine()
  -- optional parameters
  cmd:option('-seed',123,'random number generator\'s seed')
  cmd:option('-activation','ReLU','Activation')
  cmd:option('-kernelSize',3,'Size of kernel for convnet ')
  cmd:option('-pooling',0,'pooling')
  cmd:option('-batchNormalization',true,'Use batch normalization')
  cmd:option('-hiddenFeatures',{24},'Hidden features')
  cmd:text()

  local template_kwargs = cmd:parse(arg)
  template_kwargs['cuda'] = opt.cuda
  print("Options:", opt)
  print("Template Args:", template_kwargs)

  local conv_res_net = dddt.templates.conv_res_net
  local shapes = {Item=util.shape({1, 32, 32}), Stack=util.shape({1, 32, 32})}
  local dtypes = {Item=t.getdefaulttensortype(), Stack=t.getdefaulttensortype()}
  local templates = {push=conv_res_net.gen_conv_res_net,
                     pop=conv_res_net.gen_conv_res_net}

  local template_args = {push=template_kwargs, pop=template_kwargs}
  local adt, spec, constrained_types, param_funcs, interface_params = stack(shapes, dtypes, templates, template_args)

  -- Generator
  local data = require("get_mnist")()
  local generator = gen_gen(opt.batch_size, opt.cuda, opt.nitems, data)
  -- Constants: Generate constant params
  local constant_params = {empty_stack=constrained_types['Stack']:sample(t.rand, opt.cuda)}

  -- Generate interface params
  local all_params = util.update(constant_params, interface_params)

  -- load data
  -- local npy4th = require 'npy4th'
  -- local prefix = "/home/zenna/data/1467078183.7549355block_size_1__nblocks_1__nfilters_24__adt_stack__/epoch_10_run_0loss_0.187983"
  -- local push_int = npy4th.loadnpz(prefix .. "_interface_0.npz")
  -- local p = {{push_int.arr_0:cuda(), t.zeros(24):cuda()},
  --            {push_int.arr_1:cuda(), push_int.arr_2:cuda()},
  --            {push_int.arr_5:cuda(), t.zeros(1):cuda()},
  --            {push_int.arr_6:cuda(), push_int.arr_7:cuda()}}
  -- local pop_int = npy4th.loadnpz(prefix .. "_interface_1.npz")
  -- local q = {{pop_int.arr_0:cuda(), t.zeros(24):cuda()},
  --            {pop_int.arr_1:cuda(), pop_int.arr_2:cuda()},
  --            {pop_int.arr_5:cuda(), t.zeros(2):cuda()},
  --            {pop_int.arr_6:cuda(), pop_int.arr_7:cuda()}}
  -- local constant = npy4th.loadnpz(prefix .. "_constant_0.npz")
  -- local e = constant.arr_0:view(1,28,28):cuda()
  -- local all_params = {push=p, pop=q, empty_stack=e}
  train.train(param_funcs, spec.axiom, all_params, adt.constants, generator, opt)
end

main()

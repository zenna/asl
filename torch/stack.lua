local t = require "torch"
local dddt = require "types"
local util = require "util"
local res_net = require "res_net"
local gen = require "generators"
local Type = dddt.Type
local Interface = dddt.Interface
local RandVar = dddt.RandVar
local Axiom = dddt.Axiom
local AbstractDataType = dddt.AbstractDataType

nn = require "nn"


-- Example
local function stack_adt(stack_shape, item_shape, push_args, pop_args)
  local Stack = Type(stack_shape, 'Stack')
  local Item = Type(item_shape, 'Item')

  local push = Interface({Stack, Item}, {Stack}, 'push', push_args)
  local pop = Interface({Stack}, {Stack, Item}, 'pop', pop_args)
  local funcs = {push, pop}
  local consts = {}

  -- random variables
  local stack1 = RandVar(Stack, 'stack1')
  local item1 = RandVar(Item, 'item1')
  local randvars = {stack1, item1}

  -- Extensional axioms
  -- local ex_axiom = function(stack, nitems)
  --   for i = 1, nitems.get_value() do
  --     local stack = push(stack, items[i].input_var)
  --     local pop_stack = stack
  --     for j = i, 1, -1 do
  --       pop_stack, pop_item = pop(pop_stack)
  --       axiom = Axiom({pop_item}, {items[j].input_var})
  --       axioms.append(axiom)
  --     end
  --   end
  -- end

  -- Intensional Axiom
  local axiom1 = Axiom(pop:call(push:call({stack1, item1})), {stack1, item1})
  local axioms = {axiom1}
  local adt = AbstractDataType(funcs, consts, randvars, axioms, "stack")
  return adt
end

function mse(a, b)
  print(a:size(),b:size())
  local a_b = a - b
  return t.cmul(a_b, a_b):sum()
end

item_shape = util.shape({1, 32, 32})
stack_shape = util.shape({1, 50, 50})

template_kwargs = {template = res_net.nnet, gen_params = res_net.net_params}
adt = stack_adt(util.shape({10}), util.shape({10}), template_kwargs, template_kwargs)

-- Training
batchsize = 1
trainData, testData, classes = require('./get_mnist.lua')()
coroutines = {gen.infinite_samples(item_shape, t.rand, batchsize),
              gen.infinite_minibatches(trainData.x, batchsize,  true)}

-- grad = require "autograd"
-- mse = grad.nn.MSECriterion()
training = require "train"
gen.assign(adt.randvars, coroutines)

training.train(adt, 10, 5, "testing", "mysavedir")

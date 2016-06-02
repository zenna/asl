local t = require "torch"
local dddt = require "dddt"
local util = dddt.util
local res_net = dddt.templates.res_net
local gen = dddt.generators
-- local Spec = dddt.types.Spec
local Type = dddt.types.Type
local ConjAxiom = dddt.types.ConjAxiom
local EqAxiom = dddt.types.EqAxiom
local Interface = dddt.types.Interface
local RandVar = dddt.types.RandVar
local AbstractDataType = dddt.types.AbstractDataType
local DataType = dddt.types.DataType

local function stack_adt()
  local Stack = Type('Stack')
  local Item = Type('Item')
  local types = {Stack, Item}
  local push = Interface({Stack, Item}, {Stack}, 'push')
  local pop = Interface({Stack}, {Stack, Item}, 'pop')
  local funcs = {push, pop}
  local consts = {}
  stack_adt = AbstractDataType(types, funcs, consts, "Stack")
  return stack_adt
end

-- Example
local function stack_adt_axioms(stack_shape, item_shape, push_args, pop_args)
  local adt = stack_adt()
  local Stack, Item = unpack(adt.typenames)
  local pop, push = unpack(adt.interface_names)
  local data_type = DataType(stack_adt, {stack_shape, item_shape})

  -- Intensional Axiomitisation
  local stack1 = RandVar(Stack, 'stack1')
  local item1 = RandVar(Item, 'item1')
  local randvars = {stack1, item1}
  local axiom1 = EqAxiom(pop:call(push:call({stack1, item1})), {stack1, item1})
  local axiom = ConjAxiom({axiom1})
  local spec = Spec(randvars, axioms)
  return adt, data_type, axiom, spec
end

item_shape = util.shape({1, 32, 32})
stack_shape = util.shape({1, 50, 50})

item_shape = util.shape({1, 5, 5})
stack_shape = util.shape({1, 5, 5})


template_kwargs = {}
template_kwargs['layer_width'] = 10
template_kwargs['block_size'] = 2
template_kwargs['nblocks'] = 2
template_kwargs['template'] = gen_res_net
template_kwargs['template_gen'] = res_net.gen_res_net
adt = stack_adt_axioms(stack_shape, item_shape, template_kwargs, template_kwargs)

-- Training
batchsize = 2
trainData, testData, classes = require('./get_mnist.lua')()
-- coroutines = {gen.infinite_samples(stack_shape, t.rand, batchsize),
--               gen.infinite_minibatches(trainData.x:double(), batchsize,  true)}

coroutines = {gen.infinite_samples(stack_shape, t.rand, batchsize),
              gen.infinite_samples(item_shape, t.rand, batchsize)}
-- grad = require "autograd"
training = require "train"
gen.assign(adt.randvars, coroutines)

-- training.train(adt, 10, 5, "testing", "mysavedir")

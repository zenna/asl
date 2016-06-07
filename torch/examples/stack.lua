local t = require "torch"
local dddt = require "dddt"
local util = dddt.util
local map = util.map
local res_net = dddt.templates.res_net
local gen = dddt.generators
-- local Spec = dddt.types.Spec
local Type = dddt.types.Type
local ConstrainedType = dddt.types.ConstrainedType
local ConjAxiom = dddt.types.ConjAxiom
local EqAxiom = dddt.types.EqAxiom
local Interface = dddt.types.Interface
local RandVar = dddt.types.RandVar
local AbstractDataType = dddt.types.AbstractDataType
local Spec = dddt.types.Spec
local ParamFunc = dddt.types.ParamFunc
-- training = require "train"


-- Genereates the stack abstract data type
local function stack_adt()
  local Stack = Type('Stack')
  local Item = Type('Item')
  local push = Interface({Stack, Item}, {Stack}, 'push')
  local pop = Interface({Stack}, {Stack, Item}, 'pop')
  return AbstractDataType({push, pop}, {}, "Stack")
end

-- Genereates the stack specification
local function stack_spec(adt)
  local push, pop = unpack(adt.interfaces)
  -- Intensional Axiomitisation
  local stack1 = RandVar(Type('Stack'), 'stack1')
  local item1 = RandVar(Type('Item'), 'item1')
  local axiom = EqAxiom(pop:call(push:call({stack1, item1})), {stack1, item1})
  return Spec({stack1, item1}, axiom)
end

-- Generates a stack parameterised interface
local function stack_constrained(adt, stack_shape, stack_dtype, item_shape, item_dtype)
  -- Give a shape and type to Stack and Items and construct a new Data Type
  local CStack = ConstrainedType('Stack', stack_shape, stack_dtype)
  local CItem = ConstrainedType('Item', item_shape, item_dtype)
  local type_to_constrained = {Stack=CStack, Item=CItem}
  local ok = function(x) return constrain_interface(x, type_to_constrained) end
  local cinterface = map(ok, adt.interfaces)
  return AbstractDataType(cinterface, {}, adt.name)
end

-- Generates a stack parameterised interface
local function stack_param(cdt, push_args, push_template, pop_args, pop_template)
  local push, pop = unpack(cdt.interfaces)
  local push_pf = ParamFunc(push, push_template(push, push_args))
  local pop_pf = ParamFunc(pop, pop_template(pop, pop_args))
  return {push_pf, pop_pf}
end


-- Example
local function stack(stack_shape, stack_dtype, item_shape, item_dtype,
                     push_args, push_template, pop_args, pop_template)
  local adt = stack_adt()
  local spec = stack_spec(adt)
  local cdt = stack_constrained(adt, stack_shape, stack_dtype, item_shape, item_dtype)
  local pdt = stack_param(cdt, push_args, push_template, pop_args, pop_template)
  return adt, spec, cdt, pdt
end

item_shape = util.shape({1, 32, 32})
stack_shape = util.shape({1, 50, 50})
stack_dtype = torch.getdefaulttensortype()
item_dtype = torch.getdefaulttensortype()
push_template = res_net.gen_res_net
pop_template =  res_net.gen_res_net
batchsize = 2
template_kwargs = {}
template_kwargs['layer_width'] = 10
template_kwargs['block_size'] = 2
template_kwargs['nblocks'] = 2
push_args = template_kwargs
pop_args = template_kwargs
adt, spec, cdt, pdt = stack(stack_shape, stack_dtype, item_shape, item_dtype,
                            push_args, push_template, pop_args, pop_template)

-- Training
trainData, testData, classes = require('./get_mnist.lua')()
coroutines = {gen.infinite_samples(stack_shape, t.rand, batchsize),
              gen.infinite_minibatches(trainData.x:double(), batchsize,  true)}

-- grad = require "autograd"
gen.assign(spec.randvars, coroutines)
-- spec.axiom:losses(distances.mse)
-- training.train(adt, 10, 5, "testing", "mysavedir")

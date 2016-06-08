local t = require "torch"
local dddt = require "dddt"
local util = dddt.util
local map = util.map
local res_net = dddt.templates.res_net
local gen = dddt.generators
-- local Spec = dddt.types.Spec
local Type = dddt.types.Type
local ConstrainedType = dddt.types.ConstrainedType
local loss_fn = dddt.types.loss_fn
local eq_axiom = dddt.types.eq_axiom
local Interface = dddt.types.Interface
local RandVar = dddt.types.RandVar
local AbstractDataType = dddt.types.AbstractDataType
local Spec = dddt.types.Spec
local ParamFunc = dddt.types.ParamFunc
local ConcreteFunc = dddt.types.ConcreteFunc
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
  -- local axiom = EqAxiom(pop:call(push:call({stack1, item1})), {stack1, item1})
  -- An axiom is a function
  local function axiom(funcs, randvars)
    local pushy, poppy = funcs['push'], funcs['pop']
    local stacky1, itemy1 = randvars['stack1'], randvars['item1']
    local axiom2 = eq_axiom(poppy:call(pushy:call({stacky1, itemy1})), {stacky1, itemy1})
    return {axiom2}
  end
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
template_kwargs['nblocks'] = 1
push_args = template_kwargs
pop_args = template_kwargs
adt, spec, cdt, pdt = stack(stack_shape, stack_dtype, item_shape, item_dtype,
                            push_args, push_template, pop_args, pop_template)

cfs = util.mapn(function(pf, param) return ConcreteFunc.fromParamFunc(pf, param) end, pdt, params)

-- Generators
trainData, testData, classes = require('./get_mnist.lua')()
coroutines = {stack1=gen.infinite_samples(stack_shape, t.rand, batchsize),
              item1=gen.infinite_minibatches(trainData.x:double(), batchsize,  true)}

-- function sgd_update(params, delta_params, learning_rate)

function generate_randvars(randvars, coroutines)
  local randvars_samples = {}
  for k, v in pairs(randvars) do
    local coroutineok, value = coroutine.resume(coroutines[v.name])
    randvars_samples[v.name] = value
  end
  return randvars_samples
end


function train(pdt, spec, params, coroutines, num_epochs, save_every, sfx, save_dir)
  local grad = require "autograd"
  print("Starting Training")
  local loss_func = loss_fn(spec.axiom, pdt)
  local df_loss_func = grad(loss_func)
  local stats = {loss_vars = {}, loss_sums = {}}
  local optimfn, states = grad.optim.sgd(df_loss_func, state, params)
  local val_randvars = generate_randvars(spec.randvars, coroutines)
  -- print(params)
  for epoch = 1, num_epochs do
    print("Validate", loss_func(params, val_randvars))
    randvars = generate_randvars(spec.randvars, coroutines)
    -- print(params)
    -- local delta_params, loss = df_loss_func(params, randvars)
    local state = { learningRate = 0.00001 }
    params, loss = optimfn(randvars)
    print("Loss:", loss)
  end
end

params = util.map(function(pf) return pf:gen_params() end, pdt)
train(pdt, spec, params, coroutines, 10000)

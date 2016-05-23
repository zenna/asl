local t = require "torch"
local dddt = require "types"
local util = require "util"
local res_net = require "res_net"
local Type = dddt.Type
local Interface = dddt.Interface
local RandVar = dddt.RandVar
local Axiom = dddt.Axiom
local AbstractDataType = dddt.AbstractDataType

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

  -- axioms
  -- Different ways we could do it.
  -- Either we could unroll it like I did in the theano
  -- Or I just evaluate it on different stack inputs
  -- The latter approach is conceptually better
  -- But we don't want to repeat acomputation
  -- Mechanically,
  -- We want to start with the empty stack
  -- Push an item to it

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

function mse_loss(x, y)
  print("x", x)
  print("y", y)
  return x - y
end

mm = require "nn"


-- grad = require "autograd"
-- local function test()
template_kwargs = {template = res_net.nnet, gen_params = res_net.net_params}
n_adt = stack_adt(util.shape({10}), util.shape({10}), template_kwargs, template_kwargs)
-- mse = grad.nn.MSECriterion()
mse = mse_loss

-- test()


function rand_gen(shape)
  local gen = function()
    return t.rand(shape)
  end
  return gen
end

function gen_generators(randvars)
  local generators = {}
  for i = 1, #randvars do
    print(i)
    print(randvars[i].type.shape)
    table.insert(generators, rand_gen(randvars[i].type.shape))
  end
  return generators
end

function assign(adt, generators)
  for i = 1, #adt.randvars do
    adt.randvars[i].get_value = generators[i]
  end
end

generators = gen_generators(n_adt.randvars)
assign(n_adt, generators)
loss = dddt.get_losses(n_adt.axioms[1], mse)

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
  local Stack = Type(stack_shape)
  local Item = Type(item_shape)

  local push = Interface({Stack, Item}, {Stack}, 'push', push_args)
  local pop = Interface({Stack}, {Stack, Item}, 'pop', pop_args)
  local funcs = {push, pop}
  local consts = {}

  -- random variables
  local stack1 = RandVar(Stack)
  local item1 = RandVar(Item)
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
  local ex_axiom = function(stack, nitems)
    for i = 1, nitems.get_value() do
      local stack = push(stack, items[i].input_var)
      local pop_stack = stack
      for j = i, 1, -1 do
        pop_stack, pop_item = pop(pop_stack)
        axiom = Axiom({pop_item}, {items[j].input_var})
        axioms.append(axiom)
      end
    end
  end

  -- Intensional Axiom
  -- local axiom1 = Axiom(pop(push(stack1, item1)), {stack1, item1})
  -- local axioms = {axiom1}
  local axioms = {}
  local adt = AbstractDataType(funcs, consts, randvars, axioms, "stack")
  return adt
end


local function test()
  local template_kwargs = {template = res_net.nnet, gen_params = res_net.net_params}
  local n_adt = stack_adt(util.shape({10}), util.shape({10}), template_kwargs, template_kwargs)

  -- local x = t.randn(1,100)
  -- local op = n_adt.funcs[1]:call(x)
  -- print(op)
end

test()

local util = require "util"
local ConcreteFunc = require "newtypes/concretefunc".ConcreteFunc
-- local constructor = util.constructor
local distances = require "distances"
local mse = distances.mse

local module = {}
-- Helpers
local function reduce(func, tbl)
  -- assert(#tbl > 1)
  if #tbl == 1 then
    return tbl[1]
  end
  local accum = tbl[1]
  for i = 2, #tbl do
    accum = func(accum, tbl[i])
  end
  return accum
end

local function add(x, y) return x + y end
-- Axioms
---------

-- Conceptually an axiom is cleanest thought of as a function, which takes as
-- input any interfaces and returns a value in {0,1}
-- We don't literally implement an axiom like that because it would force us
-- to either have one monolothic axiom, or would prevent us from reusing
-- computation between values.

-- So there are a couple was of doing it, with an explicit cache or with
-- These sybolic rand vars.
-- The explicit cache I think is maybe simpler to implement.  Like we have a
-- an object and it maps "push(randvar)"

-- (Set of) Equational Axiom: a1 = b2, a2 = b2

function module.eq_axiom(lhs, rhs)
  local dists = util.mapn(mse, lhs, rhs)
  -- print(util.extract('value', dists))
  -- dbg()
  return reduce(add, dists)/#dists
end

function module.loss_fn(axiom, param_funcs, constants, batch_size)
  -- Loss function accepts
  return function(params, randvars)
    -- Contruct concrete functs from
    local funcs = {}
    for name, pf in pairs(param_funcs) do
      funcs[name] = ConcreteFunc.fromParamFunc(pf, params[name])
    end
    -- TODO, extrat constants from params
    local constant_vals = {}
    for k, v in pairs(constants) do
      local constant = params[k]
      local new_shape1 = util.add_batch(constant:size(), 1)
      local new_shape2 = util.add_batch(constant:size(), batch_size)
      assert(constant ~= nil)
      constant_vals[k] = constant:view(new_shape1):expand(new_shape2)
    end
    local axiom_losses = axiom(funcs, randvars, constant_vals)

    return reduce(add, axiom_losses)
  end
end

return module

local util = require "util"
local ConcreteFunc = require "newtypes/concretefunc".ConcreteFunc
-- local constructor = util.constructor
local distances = require "distances"
local mse = distances.mse

local module = {}
-- Helpers
local function reduce(func, tbl)
  assert(#tbl > 1)
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
  return reduce(add, dists)
end

function module.loss_fn(axiom, pdt)
  return function(params, randvars, opt)
    -- Contruct concrete functs from
    local funcs = {}
    for i, pf in ipairs(pdt) do
      local name = pf.interface.name
      funcs[name] = ConcreteFunc.fromParamFunc(pf, params[i])
    end
    local axiom_losses = axiom(funcs, randvars)
    return axiom_losses[1]
    -- return reduce(add, axiom_losses)
  end
end

return module

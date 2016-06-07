local util = require "util"
local ConcreteFunc = require "newtypes/concretefunc".ConcreteFunc
local constructor = util.constructor

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
local EqAxiom = {}
EqAxiom.__index = EqAxiom
function EqAxiom.new(lhs, rhs, name)
  local self = setmetatable({}, EqAxiom)
  assert(#lhs == #rhs)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  return self
end
constructor(EqAxiom)

function EqAxiom:naxioms()
  return #self.lhs
end

-- For axiom a = b return dist(a,b)
function EqAxiom:losses(dist, funcs)
  return util.mapn(function(x,y) return dist(x:gen(funcs),y:gen(funcs)) end,
                   self.lhs, self.rhs)
end

local function reduce(func, tbl)
  assert(#tbl > 1)
  local accum = tbl[1]
  for i = 2, #tbl do
    accum = func(accum, tbl[i])
  end
  return accum
end

local function add(x, y) return x + y end

function EqAxiom:losses_fn(dist, pdt)
  return function(params)
    -- Contruct concrete functs from
    local funcs = {}
    for i, pf in ipairs(pdt) do
      local name = pf.interface.name
      funcs[name] = ConcreteFunc.fromParamFunc(pf, params[i])
    end
    local losses = self:losses(dist, funcs)
    -- print("losses", losses)
    return reduce(add, losses)
  end
end

-- Conjunction of Axioms: axiom1 and axiom2 and axiom3
local ConjAxiom = {}
ConjAxiom.__index = ConjAxiom
function ConjAxiom.new(axioms)
  local self = setmetatable({}, ConjAxiom)
  self.axioms = axioms
  return self
end
constructor(ConjAxiom)

return {EqAxiom=EqAxiom, ConjAxiom=ConjAxiom}

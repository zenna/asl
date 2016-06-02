local util = require "util"
local constructor = util.constructor

-- Axioms
---------

-- (Set of) Equational Axiom: a1 = b2, a2 = b2
local EqAxiom = {}
Axiom.__index = Axiom
function Axiom.new(lhs, rhs, name)
  local self = setmetatable({}, Axiom)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  assert(#lhs == #rhs)
  return self
end
constructor(Axiom)

function Axiom:naxioms()
  return #self.lhs
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

-- Evaluation of axioms





local function apply_axioms(params)
  -- First I need to construct concrete interfaces from params
--
-- function dddt.get_losses(axiom, dist)
--   -- returns a function that can be applied to radnom variable values
--   print("axiom", axiom)
--   local losses = {}
--   for i = 1, axiom:naxioms() do
--     print("evaluating axiom: %s" %i)
--     local lhs_val = axiom.lhs[i].gen()
--     local rhs_val  = axiom.rhs[i].gen()
--     local loss = dist(lhs_val, rhs_val)
--     print("loss", loss)
--     table.insert(losses, loss)
--   end
--   return losses
-- end
--
-- function dddt.get_loss_fn(axioms, dist)
--   print("dist", dist)
--   -- Returns a loss function loss(params)
--   local loss_fn = function(params)
--     print("params here", params)
--     local losses = {}
--     for i = 1, #axioms do
--       local loss = dddt.get_losses(axioms[i], dist)
--       table.insert(losses, loss)
--     end
--     return t.Tensor(losses):sum()
--   end
--   return loss_fn
-- end

return {EqAxiom=EqAxiom, ConjAxiom=ConjAxiom}

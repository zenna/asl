local util = require "util"
local constructor = util.constructor

-- Axioms
---------

-- (Set of) Equational Axiom: a1 = b2, a2 = b2
local EqAxiom = {}
EqAxiom.__index = EqAxiom
function EqAxiom.new(lhs, rhs, name)
  local self = setmetatable({}, EqAxiom)
  self.lhs = lhs
  self.rhs = rhs
  self.name = name
  assert(#lhs == #rhs)
  return self
end
constructor(EqAxiom)

function EqAxiom:naxioms()
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

return {EqAxiom=EqAxiom, ConjAxiom=ConjAxiom}

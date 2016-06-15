local util = require "dddt.util"
local constructor = util.constructor

-- Axioms
---------

-- (Set of) Equational Axiom: a1 = b2, a2 = b2
local Spec = {}
Spec.__index = Spec
function Spec.new(randvars, axiom)
  local self = setmetatable({}, Spec)
  self.randvars = randvars
  self.axiom = axiom
  return self
end
constructor(Spec)

return {Spec=Spec}

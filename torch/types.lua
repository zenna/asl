-- Principles of the type system
--------------------------------

-- There are three main objects: (i) values (ii) types, (iii) type representations
-- Types represent sets of values. A type is some subset of the Unvierse of all values
-- There are three classes of types: Basic types, product types and function types.
--- A basic type is subset of U
--- A function type is some subset of U -> U
--- A product type of A ⊆ U is A × A
-- Types (and sets of values in general) can be represented
--- Implicitly, as a  cosntraint (e.g. constriant) on another type - A = {x \in U ; f(x)}
--- Explicitly, as an enumeration of values in the type A = {(a1, a2, a3, ...)}
--- Parametrically, as a function from some space to elements of the type A : Θ -> U
-- A specification is again a restriction over types

-- This is not quite right, theres a difference between two types which represent the same
-- set of values.  Like an immutable Image x::Float64 end is different from Float64.

util = require("util")

local types = {}
types = util.update(types, require("newtypes/type"))
types = util.update(types, require("newtypes/interface"))
types = util.update(types, require("newtypes/constant"))
types = util.update(types, require("newtypes/interface"))
types = util.update(types, require("newtypes/datatypes"))
types = util.update(types, require("newtypes/randvar"))
types = util.update(types, require("newtypes/axioms"))
types = util.update(types, require("newtypes/spec"))
types = util.update(types, require("newtypes/paramfunc"))
types = util.update(types, require("newtypes/concretefunc"))


return types

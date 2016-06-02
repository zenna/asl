local util = require "util"
local types = {}
types = util.update(types, require("newtypes/type"))
types = util.update(types, require("newtypes/interface"))
types = util.update(types, require("newtypes/constant"))
types = util.update(types, require("newtypes/interface"))
types = util.update(types, require("newtypes/datatypes"))
types = util.update(types, require("newtypes/randvar"))


return types
-- Applying functions to random variables

-- If the function is parameterised the result must be a paramterised
-- random variables.  That is, in order to get a concrete value you need to
-- provide values for teh paramters

-- Alternatively.  push(stack) is only valid when push is concrete.
-- That is we make the axioms a function of values and pass in a concrete function.
-- The problem I had with this was that g
-- well firs toff whats the output of push.  It could well be a symbolic
-- random variable
-- -- The only issue is use of values between axioms
-- push({stack, item1})
-- function(pop, push)
--   return EqAxiom(pop(push({stack1, item1})), {stack1, item1})
-- end
--
--
--
-- -- Whats the diference between Stack and empty stack.
-- -- Stack is an unrestricted space of things. Constant is a value of that type
--
-- return {Interface=Interface, AbstractDataType=AbstractDataType}
-- We need a simple implementation of a concrete data type and a separation
-- of concerns
-- An interface should specify: the types of the input and output
-- There's an assignment of a function space
-- And then there's an assignment of a function
-- Currently it also specifies a function space

-- Types
-- Type: A data type.  Pretty much just a name
-- Interface: a functional type specification for a name
-- Constant: a functional type specification for a constant
-- ADT = ({T}, {O}, {F})

-- Should a type contain shape information?

-- Then we need something which assigns a function space to an face
-- and a value space to constants

-- Then we need concrete implementation

local util = require "util"
local constructor = util.constructor

-- Random Variable
------------------
local RandVar = {}
RandVar.__index = RandVar

function RandVar.new(type, name)
  local self = setmetatable({}, RandVar)
  self.type = type
  self.name = name
  -- self.is_stale = true
  self.gen = function()
    error("No distribution assigned to $s::$s" % {self.type.name, self.name})
  end
  return self
end
constructor(RandVar)

-- function RandVar:value()
--   if self.is_stable then
--     error("Pulling stale value")
--   else
--     return self.val
--   end
-- end
--
-- function RandVar:set_value(val)
--   self.val = val
--   self.is_stable = false
-- end

-- function RandVar:set_params(params)
--   self.params = params
-- end

-- A transformed randvar is one thats been modifed by an un
local TransformedRandVar = {}
TransformedRandVar.__index = TransformedRandVar

function TransformedRandVar.new(interface, inp_randvars, out_index)
  local self = setmetatable({}, TransformedRandVar)
  self.interface = interface
  self.inp_randvars = inp_randvars
  self.out_index = out_index
  self.type = self.interface.rhs[out_index]
  return self
end
constructor(TransformedRandVar)

function TransformedRandVar:gen(funcs)
  local func = funcs[self.interface.name]
  assert(func ~= nil)
  local values = util.map(function(rv) return rv:gen(funcs) end, self.inp_randvars)
  return self.func:call(values)[index]
end


-- ParamRandVar : f(X) where X is a randvar and f is a paramterised function
-- local ParamRandVar = {}
-- ParamRandVar.__index = ParamRandVar
--
-- function ParamRandVar.new(type, inp_randvars, index, func)
--   local self = setmetatable({}, ParamRandVar)
--   self.type = type
--   self.func = func
--   self.inp_randvars = inp_randvars
--   self.index = index
--   self.params = nil
--   return self
-- end
-- constructor(ParamRandVar)
--
-- function ParamRandVar:gen()
--   local values = {}
--   for i , inp_randvar in ipairs(self.inp_randvars) do
--     table.insert(values, inp_randvar:gen())
--   end
--   return self.func:call(values, self:params())[index]
-- end
--
-- function ParamRandVar:params()
--   if self.paramsa == nil then
--     error("Parameters not set")
--   else
--     return self.params
--   end
-- end

function RandVar:set_params(params)
  self.params = params
end


-- function ParameterisedRandVar:gen_gen(inp_randvars, i)
--   local gen = function()
--     -- Check whether any of my inputs have changed (if not I needn't change)
--     local inps_changed = false
--     for j = 1, #inp_randvars do
--       inps_changed = inps_changed or inp_randvars[j].is_stale
--     end
--     -- Only recompute if I'm not initialised (is_stale) or inputs have changed
--     if inps_changed or r:is_stale() then
--       local inp_randvars_vars = {}
--       for j = 1, #inp_randvars do
--         local q = inp_randvars[j].gen()
--         table.insert(inp_randvars_vars, q)
--       end
--       local val = self.template(inp_randvars_vars, r:params())[i]
--       r:set_value(val)
--       print("Inputs changed, recomputed output")
--       return val
--     else
--       print("Using Cache")
--       return r:value()
--     end
--   end
--   return gen
-- end


return {RandVar=RandVar, TransformedRandVar=TransformedRandVar}

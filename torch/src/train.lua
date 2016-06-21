local training = {}
local grad = require "autograd"
local gradcheck = require 'autograd.gradcheck' {randomizeInput = false}
local loss_fn = require("dddt.types.axioms").loss_fn
local torch = require("torch")
local util = require("dddt.util")

local function compute_stats(tensor, statistics)
  return util.map(function (s) return s(tensor) end, statistics)
end

local function paramwalk(params, statistics)
  local output = {}
  for k,v in pairs(params) do
    if torch.isTensor(v) then
      output[k] = compute_stats(v, statistics)
    else
      output[k] = paramwalk(v, statistics)
    end
  end
  return output
end

local function sizetostring(s)
  local str = "{"
  for i=1,s:size() do
    str = str .. s[i] .. " "
  end
  return str .. "}"
end
local function size(x) return sizetostring(x:size()) end

function training.train(param_funcs, axiom, params, constants, generator, opt)
  local stats_every = opt['stats_every'] or 100
  local grad_check = opt['grad_check'] or true

  local loss_func = loss_fn(axiom, param_funcs, constants, opt.batch_size)
  local df_loss_func = grad(loss_func)
  local val_randvars = generator()
  local grads, loss = df_loss_func(params, val_randvars)
  print("Initial Parameters")
  print(paramwalk(params, {mean=torch.mean, var=torch.var, size=size}))
  print("Initial Gradients")
  print(paramwalk(grads, {mean=torch.mean, var=torch.var, size=size}))

  -- debug

  -- local grads, val = df_loss_func(params, val_randvars)
  -- print("val", val)
  -- print(paramwalk({e=grads.empty_stack}, {mean=torch.mean, var=torch.var, size=size}))
  -- print(paramwalk({e=params.empty_stack}, {mean=torch.mean, var=torch.var, size=size}))
  -- local grads, val2 = df_loss_func(params, val_randvars)
  -- print("val2", val2)
  -- print(paramwalk({e=grads.empty_stack}, {mean=torch.mean, var=torch.var, size=size}))
  -- print(paramwalk({e=params.empty_stack}, {mean=torch.mean, var=torch.var, size=size}))
  -- print("diff", val - val2)
  -- if grad_check then
  --   print("Gradient Check")
  --   -- dbg()
  --   assert(gradcheck(loss_func, params, val_randvars),
  --          'incorrect gradients on W and x')
  -- end
  local stats = {loss_vars = {}, loss_sums = {}}
  local optimfn, states = opt.optim_alg(df_loss_func, opt.optim_state, params)
  print("Validate", loss_func(params, val_randvars))

  print("Starting Training")
  for epoch = 1, opt.num_epochs do
    -- print("Validate", loss_func(params, val_randvars))
    local randvars = generator()
    -- local grads, loss = df_loss_func(params, randvars)
    local grads, loss = optimfn(randvars)
    print("epoch:", epoch, "Loss:", loss)
    if epoch % stats_every == 1 then
      print("Parameters")
      print(paramwalk(params, {mean=torch.mean, var=torch.var, size=size}))
      print("Gradients")
      print(paramwalk(grads, {mean=torch.mean, var=torch.var, size=size}))
    end
  end
end

return training

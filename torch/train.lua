local training = {}
local grad = require "autograd"
local loss_fn = require("newtypes/axioms").loss_fn


local function generate_randvars(randvars, coroutines)
  local randvars_samples = {}
  for k, v in pairs(randvars) do
    local coroutineok, value = coroutine.resume(coroutines[v.name])
    randvars_samples[v.name] = value
  end
  return randvars_samples
end


function training.train(pdt, spec, params, coroutines, num_epochs, save_every, sfx, save_dir)
  print("Starting Training")
  local loss_func = loss_fn(spec.axiom, pdt)
  local df_loss_func = grad(loss_func)
  -- local stats = {loss_vars = {}, loss_sums = {}}
  local state = { learningRate = 0.00001 }
  local optimfn, states = grad.optim.sgd(df_loss_func, state, params)
  local val_randvars = generate_randvars(spec.randvars, coroutines)
  -- print(params)
  for epoch = 1, num_epochs do
    print("Validate", loss_func(params, val_randvars))
    local randvars = generate_randvars(spec.randvars, coroutines)
    -- print(params)
    -- local delta_params, loss = df_loss_func(params, randvars)
    local params, loss = optimfn(randvars)
    print("Loss:", loss)
  end
end

return training

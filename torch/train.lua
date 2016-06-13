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


function training.train(param_funcs, axiom, params, constants, generator,
                        batch_size, num_epochs, save_every, sfx, save_dir)
  print("Starting Training")
  print("CCC", constants)
  local loss_func = loss_fn(axiom, param_funcs, constants, batch_size)
  local df_loss_func = grad(loss_func)
  -- local stats = {loss_vars = {}, loss_sums = {}}
  local state = { learningRate = 0.001 }
  local optimfn, states = grad.optim.adam(df_loss_func, state, params)
  local val_randvars = generator()
  -- print(params)
  for epoch = 1, num_epochs do
    -- print("Validate", loss_func(params, val_randvars))
    if epoch > 310 then
      -- dbg()
    end
    local randvars = generator()
    local params, loss = optimfn(randvars)
    print("epoch:", epoch, "Loss:", loss)
  end
end

return training

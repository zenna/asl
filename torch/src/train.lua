local training = {}
local grad = require "autograd"
local loss_fn = require("dddt.types.axioms").loss_fn

function training.train(param_funcs, axiom, params, constants, generator, opt)
  print("Starting Training")
  local loss_func = loss_fn(axiom, param_funcs, constants, opt.batch_size)
  local df_loss_func = grad(loss_func)
  -- local stats = {loss_vars = {}, loss_sums = {}}
  local optimfn, states = opt.optim_alg(df_loss_func, opt.optim_state, params)
  local val_randvars = generator()
  for epoch = 1, opt.num_epochs do
    -- print("Validate", loss_func(params, val_randvars))
    local randvars = generator()
    local params, loss = optimfn(randvars)
    print("epoch:", epoch, "Loss:", loss)
  end
end

return training

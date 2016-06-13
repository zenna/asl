local gen = {}
local t = require("torch")
local util = require("util")

local function wrap(co)
  return function()
    local co_is_good, val = coroutine.resume(co)
    assert(co_is_good, "coroutine fail %s" % val)
    print("Generating", torch.sum(val))
    return val
  end
end

function gen.assign(randvars, coroutines)
  assert(#randvars == #coroutines)
  for i = 1, #randvars do
    randvars[i].gen = wrap(coroutines[i])
  end
end

function gen.infinite_samples(shape, sampler, batch_size)
  local batched_shape = util.add_batch(shape, batch_size)
  local co = coroutine.create(function()
    while true do
      coroutine.yield(sampler(batched_shape))
    end
  end)
  return co
end

function gen.infinite_minibatches(inputs, batch_size, shuffle)
  local start_idx = 1
  local nelements = inputs:size()[1]
  local indices
  if shuffle then
    indices = t.randperm(nelements):long()
  else
    indices = t.range(1, nelements):long()
  end
  local co = coroutine.create(function ()
    while true do
      local batch_indices = util.circular_indices(1, nelements, start_idx, batch_size)
      start_idx = batch_indices[-1] + 1
      if start_idx > nelements then
        start_idx = 1
      end
      if shuffle then
        batch_indices = indices:index(1, batch_indices)
      end
      local minibatch = inputs:index(1, batch_indices)
      coroutine.yield(minibatch)
    end
  end)
  return co
end

return gen

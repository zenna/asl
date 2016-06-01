local gen = {}
local t = require("torch")
local util = require("util")

local function wrap(co)
  return function()
    local co_is_good, val = coroutine.resume(co)
    assert(co_is_good, "coroutine fail %s" % val)
    return val
  end
end

function gen.assign(randvars, coroutines)
  assert(#randvars == #coroutines)
  for i = 1, #randvars do
    randvars[i].gen = wrap(coroutines[i])
  end
end

function gen.infinite_samples(shape, sampler, batchsize)
  local batched_shape = util.add_batch(shape, batchsize)
  local co = coroutine.create(function()
    while true do
      coroutine.yield(sampler(batched_shape))
    end
  end)
  return co
end

function gen.infinite_minibatches(inputs, batchsize, shuffle)
  local start_idx = 1
  local nelements = inputs:size()[1]
  local indices
  if shuffle then
    indices = t.randperm(nelements):long()
  else
    indices = t.range(1, nelements):long()
  end
  local co = coroutine.create(function ()
    local end_idx
    while true do
      end_idx = start_idx + batchsize
      local batch_indices = util.circular_indices(start_idx, end_idx, nelements)
      if shuffle then
        batch_indices = indices:index(1, batch_indices)
      end
      local minibatch = inputs:index(1, batch_indices)
      coroutine.yield(minibatch)
      start_idx = end_idx
    end
  end)
  return co
end

return gen

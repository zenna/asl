local util = require 'autograd.util'

-- Get/create dataset
local function get_traindata()
   if not path.exists(sys.fpath()..'/mnist') then
      os.execute[[
      curl https://s3.amazonaws.com/torch.data/mnist.tgz -o mnist.tgz
      tar xvf mnist.tgz
      rm mnist.tgz
      ]]
   end

   local trainData = torch.load(sys.fpath()..'/mnist/train.t7')
   local data = trainData.x:type(torch.Tensor():type())
   return data:mul(1/data:max())
end

return get_traindata

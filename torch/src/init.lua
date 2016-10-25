-- Base package
local pdt = {}

-- Meta info
pdt.VERSION = '0.1'
pdt.LICENSE = 'MIT'


hascunn, cunn = pcall(require, 'cunn')
hascutorch, cutorch = pcall(require, 'cutorch')
hasdbg, dbg = pcall(require, 'debugger')

-- Sub packages:
pdt.util = require "pdt.util"
pdt.templates = require "pdt.templates"
pdt.types = require "pdt.types"
pdt.generators = require "pdt.generators"
pdt.distances = require "pdt.distances"
pdt.train = require "pdt.train"

-- require "cunn"
-- if not cutorch then
--    require 'cutorch'
--    runtests = true
-- end
--

return pdt

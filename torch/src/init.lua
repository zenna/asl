-- Base package
local dddt = {}

-- Meta info
dddt.VERSION = '0.1'
dddt.LICENSE = 'MIT'


hascunn, cunn = pcall(require, 'cunn')
hascutorch, cutorch = pcall(require, 'cutorch')
hasdbg, dbg = pcall(require, 'debugger')

-- Sub packages:
dddt.util = require "dddt.util"
dddt.templates = require "dddt.templates"
dddt.types = require "dddt.types"
dddt.generators = require "dddt.generators"
dddt.distances = require "dddt.distances"
dddt.train = require "dddt.train"

-- require "cunn"
-- if not cutorch then
--    require 'cutorch'
--    runtests = true
-- end
--

return dddt

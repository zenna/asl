-- Base package
local dddt = {}

-- Meta info
dddt.VERSION = '0.1'
dddt.LICENSE = 'MIT'

-- Sub packages:
dddt.util = require "dddt.util"
-- dddt.templates = require "dddt.templates"
dddt.types = require "dddt.types"
dddt.generators = require "generators"
dbg = require "debugger"

return dddt

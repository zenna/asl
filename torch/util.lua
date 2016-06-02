local t = require "torch"
local util = {}

-- Type Stuff
function util.constructor(atype)
  setmetatable(atype, {
    __call = function (cls, ...)
      return cls.new(...)
    end,
  })
end

-- Torch Stuff
function util.shape(shape_tbl)
  return t.LongStorage(shape_tbl)
end

function util.circular_indices(lb, ub, thresh)
  local indices
  local curr_lb = lb
  local stop
  local i = 1
  while true do
    stop = math.min(ub, thresh)
    print(curr_lb, stop, ub)
    local ix = t.range(curr_lb, stop):long()
    if i == 1 then -- ew
      indices = ix
    else
      indices = t.cat(indices, ix)
    end
    i = i + 1
    if stop ~= ub then
      local diff = ub - stop
      curr_lb = lb
      ub = diff + lb - 1
    else
      break
    end
  end
  return indices, stop
end

-- function util.add_batch(shape, batchsize)
--   local new_shp = t.LongStorage(#shape + 1)
--   new_shp[1] = batchsize
--   for i = 2, #new_shp do
--     new_shp[i] = shape[i-1]
--   end
--   return new_shp
-- end

function util.add_batch(shape, batch_size)
  return t.cat(t.LongTensor({batch_size}), t.LongTensor(shape)):storage()
end


-- Fun little functions
function util.identity(x)
  return x
end

function util.get(tbl, key)
  return tbl[key]
end

function util.dotget(tbl, key)
  return tbl.key
end

-- Are all the values in this table unique
function util.all_unique(tbl)
  local keys = {}
  for i, v in ipairs(tbl) do
    if keys[v] ~= nil then
      return false
    end
    keys[v] = 1
  end
  return true
end

-- Check whether a set of values which have field type
function util.all_eq(a, b)
  assert(#a == #b, "%s ~= %s" % {#a, #b})
  for i = 1, #a do
    if a[i] ~= a[i] then
      return false
    end
  end
  return true
end

-- Functional Stuff
-------------------


function util.map(func, array)
  local new_array = {}
  for i,v in ipairs(array) do
    new_array[i] = func(v)
  end
  return new_array
end

-- functional update
function util.update(tbl1, tbl2)
  local new_tbl = {}
  for key, value in pairs(tbl1) do
    new_tbl[key] = value
  end
  for key, value in pairs(tbl2) do
    new_tbl[key] = value
  end
  return new_tbl
end

function util.mapn(func, ...)
  local new_array = {}
  local i=1
  local arg_length = table.getn(arg)
  while true do
    local arg_lis t = map(function(arr) return arr[i] end, arg)
    if table.getn(arg_list) < arg_length then return new_array end
    new_array[i] = func(unpack(arg_list))
    i = i+1
  end
end

function util.lua_remove_if(func, arr)
  local new_array = {}
  for _,v in arr do
    if not func(v) then table.insert(new_array, v) end
  end
  return new_array
end

-- String
function util.split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

-- From a list of tables contruct a table which extracts keys
function util.extract(k, xs)
  return util.map(function(x) return x[k] end, xs)
end

-- Table
function util.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and util.tostring( v ) or
      tostring( v )
  end
end

function util.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. util.val_to_str( k ) .. "]"
  end
end

function util.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, util.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        util.key_to_str( k ) .. "=" .. util.val_to_str( v ) )
    end
  end
  return table.concat( result, "," )
end


return util

logstate = {}

def log(name, value):
  logstate[name] = value
  return value

def log_append(name, value):
  if name in logstate:
    logstate[name].append(value)
  else:
    logstate[name] = [value]
  return value


def reset_log():
  global logstate
  logstate.clear()


def getlog():
  return logstate

logstate = {}

def log(name, value):
  logstate[name] = value

def log_append(name, value):
  if name in logstate:
    logstate[name].append(value)
  else:
    logstate[name] = [value]


def reset_log():
  global logstate
  logstate.clear()


def getlog():
  return logstate

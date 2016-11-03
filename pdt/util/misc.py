
# Primitive Functions
def upper_div(x, y):
    a, b = divmod(x, y)
    if b == 0:
        return a
    else:
        return a + 1


def identity(x):
    return x


def upper_div(x, y):
    a, b = divmod(x, y)
    if b == 0:
        return a
    else:
        return a + 1


# Dict Functions
def stringy(ls):
    out = ""
    for l in ls:
        out = out + str(l) + "_"
    return out


def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s_%s__" % (str(key), str(val))
    return out

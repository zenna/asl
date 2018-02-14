def dicttracegen(nitems):
  print("Making dict trace with {} items".format(nitems))
  def dicttrace(items, runstate, set_item, get_item, empty):
    """Example dict trace"""
    asl.log_append("empty", empty)
    adict = empty
    # keyset = [next(items) for i in range(nitems)]
    k1 = next(items)
    k2 = next(items)
    v1 = next(items)
    v2 = next(items)
    asl.log_append("{}/internal".format(runstate['mode']), v1)
    asl.log_append("{}/internal".format(runstate['mode']), v2)

    (adict,) = set_item(adict, k1, v1)
    asl.log_append("{}/internal".format(runstate['mode']), adict)

    (adict,) = set_item(adict, k2, v2)
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v1x, ) = get_item(adict, k1)
    (v2x, ) = get_item(adict, k2)
    asl.observe(v1x, "val1", runstate)
    asl.observe(v2x, "val2", runstate)
    (adict,) = set_item(adict, k2, v1)
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v1xx, ) = get_item(adict, k1)
    (v2xx, ) = get_item(adict, k2)
    asl.observe(v1xx, "val1_2", runstate)
    asl.observe(v2xx, "val2_2", runstate)
    return v2xx

  return dicttrace

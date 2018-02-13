def tracegen(nitems, nrounds):
  print("Making stack trace with {} items and {} rounds".format(nitems, nrounds))
  def trace(items, r, runstate, push, pop, empty):
    """Example stack trace"""
    # import pdb; pdb.set_trace()
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      for i in range(nitems):
        (stack,) = push(stack, next(items))
        # print("BLIP!")
        asl.log_append("{}/internal".format(runstate['mode']), stack)

      for j in range(nitems):
        (stack, pop_item) = pop(stack)
        asl.observe(pop_item, "pop.{}.{}".format(nr, j), runstate)
        # print("BLIP!")
        asl.log_append("{}/internal".format(runstate['mode']), stack)
      
    return pop_item
  
  return trace

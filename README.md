# Algebraic Structure Learning


[![Build Status](https://travis-ci.org/zenna/asl.svg?branch=master)](https://travis-ci.org/zenna/asl)
[![Coverage Status](https://coveralls.io/repos/github/zenna/asl/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/zenna/asl?branch=master)

asl is a framework for synthesizing approximate data structures and algorithms

### Usage

One thing you can do in asl is learn "neural" implementations data structures, like (Lists, Stacks, Dicts) using python data structures as a guide.

In this framework, a data structure has some internal state that is opaque in the sense that it is innaccessible, except for by a a set of interfaces.

For example a `Stack` has at least two operations: `push`, `pop`.

`push` is a pure function, it has the following type

`push: Stack × Item -> Stack`

First we will represent stacks with python list, and so the empty stack is just the empty list

`empty = []`

```python
def list_push(stack, element):
  stack = stack.copy()
  stack.append(element)
  return (stack, )
```

Noticed we copied that stack data structure otherwise the stack we put in would be modified and `list_push` would not be a pure function.

`pop`, the function which removes an element from a stack has a similarly straightfoward definition.

```python
def list_pop(stack):
  stack = stack.copy()
  item = stack.pop()
  return (stack, item)
```

Then we simply bundle these together in a dict

```
return {"push": list_push, "pop": list_pop, "empty": []}
```

## Neural Implementation

Now we can construct a neural implementation.
First we create a classes corresponding to each function

```python
class Push(Function):
  "Push Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Push, self).__init__([stack_type, item_type], [stack_type])

class Pop(Function):
  "Pop Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Pop, self).__init__([stack_type], [stack_type, item_type])
```

Then a neural implementation of each


## Training program
The final part is to write a reference.  The neural data structure will try to emulate the behaviour of the reference data structure in the context of the reference program.


```python
class StackSketch(Sketch):
  def sketch(self, items, push, pop, empty):
    """Example stack trace"""
    log_append("empty", empty)
    stack = empty
    (stack,) = push(stack, next(items))
    (stack,) = push(stack, next(items))
    (pop_stack, pop_item) = pop(stack)
    self.observe(pop_item)
    (pop_stack, pop_item) = pop(pop_stack)
    self.observe(pop_item)
    return pop_item
```

What do we mean by emulate?  The neural program should attempts to ensure that *observed* values are the same.
Observed values are ones which are observed with `self.observe`.
In general, not all types are observable, for instance the Stack type is not observable; it would not be meaningful to compare the python empty stack to our neural implementation.
The solution is to suggest some types are observable and others not, and strive for equivalent behaviour on observable types.

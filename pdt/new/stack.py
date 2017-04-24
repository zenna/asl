"""Stack Data structure"""
from typo import Type, ForAllVar, Function, Constant, EqAxiom, Method
from rewrite import rewrite
import numpy as np

# Example
# =======
def stack_adt():
    """Create an abstract stack data type"""
    Stack = Type("Stack")
    Item = Type("Item")
    s = ForAllVar(Stack, "s")
    i = ForAllVar(Item, "i")
    push = Function((Stack, Item), (Stack, ), "push")
    pop = Function((Stack, ), (Stack, Item), "pop")
    EMPTY_STACK = Constant(Stack, "EMPTY_STACK")
    pop_appl = pop(push(s, i)[0])
    s2, i2 = pop_appl[0], pop_appl[1]
    axiom1 = EqAxiom(s2, s, "stack_equivalence")
    axiom2 = EqAxiom(i2, i, "item_equivalence")
    return {'types': [Stack, Item],
            'functions': [push, pop],
            'constants': [EMPTY_STACK],
            'axioms': [axiom1, axiom2]}


def pop_once(EMPTY_STACK, push, pop, items, max_pushes=5):
    """Example interaction distribution. Pushes n times, Pops once."""
    num_pushes = np.random.randint(max_pushes)
    stack = EMPTY_STACK
    for i in range(num_pushes):
        (stack,) = push(stack, items[i])

    num_pops = np.random.randint(num_pushes)
    for i in range(num_pops):
        (stack, item) = pop(stack)

    observables = [item]
    return observables

# ## Concrete Data Type
# ## ==================
# def stack_cdt(types, functions: Sequence[Function]):
#   """Create a concrete stack data type"""
#   Stack, Item = types
#   ConcreteStack = ConcreteType(Stack, shape=(14, 14, 1), dtype=floatX())
#   ConcreteItem = ConcreteType(Item, shape=(28, 28, 1), dtype=floatX())
#
#   # Functions
#   push, pop = functions
#   concrete_push = ConcreteFunction(push, pytorch_func=...)
#   concrete_pop = ConcreteFunction(pop, pytorch_func=...)
#
#   # Constants
#   # TODO: initialize empty stack
#   ConcreteEmpty = ConcreteConstant(EMPTY_STACK)


def python_stack_cdt(push, pop):
  """Create a concrete data-structure using python list"""
  def python_pop(stack):
    stack = stack.copy()
    return (stack.pop(),)

  def python_push(stack, item):
    stack = stack.copy()
    stack.append(item)
    return (stack,)

  PYTHON_EMPTY_STACK = []
  python_push = Method(push, call=python_push)
  python_pop = Method(pop, call=python_push)
  return [python_push, python_pop]


def gen_loss_terms():
  # 1. Get symbolic execution of data-structure
  observables = pop_once(EMPTY_STACK, push, pop, items)

  # 2. From axioms, rewrite into predicates
  predicates = rewrite(observables, axioms)

  # 3. convert the predicates to a loss function

  # 4. Bind methods to symbolic functions and activate

def main():
    # Generate the symbolic data type
    stack_types, stack_functions, stack_constants, stack_axioms = getn(stack_adt(), "types", "functions", "constants", "axioms")
    observables = pop_once(EMPTY_STACK, push, pop, items)
    c_types, c_functions, c_constants = stack_cdt(types, functions, constants)

if __name__ == "__main__":
  main()

  # TODO: Something connecting the implementation to the abstract structure
  # - Is there complete independence between the calls and the data
  # - i.e. are they completely decomposable
  # TODO: 1. Generate abstract calls
  # - In some situtaitons do i need to work directly with concrete structure
  # - e.g. arcade game, isn't that easier.
  # TODO: 2. Convert that into the loss terms
  # - may have no axiosm
  # TODO: 3. Take a gradient step

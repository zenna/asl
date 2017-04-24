"""Symbolic Manipulation"""
from typing import Sequence
from typo import EqAxiom


def rewrite(observables, axioms: Sequence[EqAxiom]):
  """Transforms function applications and axioms to equations
  Args:
    observables: sequence of function applications
      e.g. [push(push(EMPTY_STACK, i1), i2) ... ]
    axioms: equational axioms for functions in observables
  """
  pass

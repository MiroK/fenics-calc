from xcalc.interpreter import Eval
from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) xcalc.interpreter (no timeseries)'''
    def test_maybe_fix_future(self):

        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, 'RT', 1)

        x = Function(V)
        y = Function(V)

        for f in (inner(x, y), as_vector((x[0], y[1])), 2*x+y):
            with self.assertRaises(AssertionError):
                Eval(f)

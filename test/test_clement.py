from xcalc.interpreter import Eval
from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestClement(unittest.TestCase):
    '''Sanity'''
    def test(self):
        errors, hs = [], []

        for n in (4, 8, 16, 32):
            mesh = UnitSquareMesh(n, n)
            x, y = SpatialCoordinate(mesh)

            uh = Eval(grad(x**2 + y**2))
            u = as_vector((2*x, 2*y))

            errors.append(error(u, uh))
            hs.append(mesh.hmin())
        self.assertTrue(np.all(np.diff(errors) < 0))
        # Actual rate
        deg = np.round(np.polyfit(np.log(hs), np.log(errors), 1)[0], 0)
        self.assertTrue(deg >= 1)




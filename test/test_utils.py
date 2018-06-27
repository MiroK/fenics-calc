from xcalc.utils import (space_of, common_sub_element, coefs_of, numpy_op_indices,
                         find_first, find_last, clip_index)
from dolfin import *
import numpy as np
import unittest


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) xcalc.utils'''
    def test_spaces_of_ok(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)
        
        V_ = space_of((f, ))

        self.assertEqual(mesh.id(), V_.mesh().id())
        self.assertEqual(V.ufl_element(), V_.ufl_element())

    def test_spaces_of_fail(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)

        V = FunctionSpace(mesh, 'CG', 2)
        g = Function(V)

        with self.assertRaises(AssertionError):
            space_of((f, g))

    def test_common_sub_element_ok(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        
        self.assertEqual(V.ufl_element(), common_sub_element((V, )))

        p = FiniteElement('Lagrange', triangle, 1)
        V = VectorFunctionSpace(mesh, 'CG', 1)
        X = FunctionSpace(mesh, MixedElement([p, p]))

        self.assertEqual(p, common_sub_element((V, X)))


    def test_common_sub_element_fail_mixed(self):
        mesh = UnitSquareMesh(3, 3)
        p = FiniteElement('Lagrange', triangle, 1)
        q = FiniteElement('Lagrange', triangle, 2)
        X = FunctionSpace(mesh, MixedElement([p, q]))

        with self.assertRaises(ValueError):
            common_sub_element((X, ))

    def test_common_sub_element_fail_no_common(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, 'CG', 1)
        W = VectorFunctionSpace(mesh, 'CG', 2)

        with self.assertRaises(AssertionError):
            common_sub_element((V, W))

    def test_coef_ok(self):
        a = 1
        self.assertEqual(coefs_of(a), a)

    def test_coef_fail(self):
        a = Constant(1)

        with self.assertRaises(AssertionError):
            coefs_of(a)

    def test_indices(self):
        mesh = UnitSquareMesh(3, 3)
        W = VectorFunctionSpace(mesh, 'CG', 2)

        true = np.column_stack([W.sub(i).dofmap().dofs() for i in range(2)])
        
        me = np.zeros_like(true)
        for row, row_values in enumerate(numpy_op_indices(W, (2, ))):
            me[row] = row_values

        error = np.linalg.norm(me - true)
        self.assertEqual(error, 0)

    def test_clipping(self):
        a = (-2, -1, 0, 2, 3, 5, 10, 12, 14, 23, 49, 65, 79)

        self.assertTrue(find_first(a, lambda x: x < 0) == 0)
        self.assertTrue(find_first(a, lambda x: x > 0 and x % 2 == 1) == 4)

        self.assertTrue(find_last(a, lambda x: x == 10) == -7)
        self.assertTrue(find_last(a, lambda x: x > 0) == -1)

        f, l = 2, 20
        i = clip_index(a, f, l)
        self.assertTrue(all(f < x < l for x in a[i]))

        

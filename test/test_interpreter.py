from xcalc.interpreter import Eval
from dolfin import *
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) xcalc.interpreter (no timeseries)'''
    def test_sanity0(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, 'CG', 1)

        f = Expression('x[0]', degree=1)
        g = Expression('x[1]', degree=1)
        a = 3
        b = -2

        u = interpolate(f, V)
        v = interpolate(g, V)

        expr = a*u + b*v

        me = Eval(expr)
        true = Expression('a*f+b*g', f=f, g=g, a=a, b=b, degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)
        
    def test_sanity1(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'DG', 0)

        u = interpolate(Expression((('x[0]', 'x[1]'),
                                    ('2*x[0]+x[1]', 'x[0]+3*x[1]')), degree=1), T)
        expr = sym(u) + skew(u)
        me = Eval(expr)
        true = u

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity2(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)

        A = interpolate(Expression((('x[0]', 'x[1]'),
                                    ('2*x[0]+x[1]', 'x[0]+3*x[1]')), degree=1), T)
        expr = tr(sym(A) + skew(A))
        me = Eval(expr)
        true = Expression('x[0] + x[0] + 3*x[1]', degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)
        
    def test_sanity3(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)

        A = interpolate(Expression((('x[0]', 'x[1]'),
                                    ('2*x[0]+x[1]', 'x[0]+3*x[1]')), degree=1), T)
        expr = (sym(A) + skew(A))[0, 0]
        me = Eval(expr)
        true = Expression('x[0]', degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity4(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)

        A = interpolate(Expression((('x[0]', 'x[1]'),
                                    ('2*x[0]+x[1]', 'x[0]+3*x[1]')), degree=1), T)
        expr = (sym(A) + skew(A))[:, 0]
        me = Eval(expr)
        true = Expression(('x[0]', '2*x[0]+x[1]'), degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity5(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)

        A = interpolate(Expression((('1', 'x[0]'),
                                    ('2', 'x[1]')), degree=1), T)
        expr = det(A)
        me = Eval(expr)
        true = Expression('x[1]-2*x[0]', degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity6(self):
        mesh = UnitCubeMesh(5, 5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)

        A = interpolate(Expression((('x[0]', '0', '1'),
                                    ('0', '1', 'x[1]'),
                                    ('x[2]', '0', '1')), degree=1), T)
        expr = det(A)
        me = Eval(expr)
        true = Expression('x[0]-x[2]', degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity7(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)
        A = interpolate(Expression((('1', 'x[0]'),
                                    ('2', 'x[1]')), degree=1), T)

        V = VectorFunctionSpace(mesh, 'CG', 1)
        v = interpolate(Expression(('x[0]+x[1]', '1'), degree=1), V)

        me = Eval(dot(A, v))
        true = Expression(('x[1]+2*x[0]', '2*x[0]+3*x[1]'), degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)

    def test_sanity8(self):
        mesh = UnitSquareMesh(5, 5)
        T = TensorFunctionSpace(mesh, 'CG', 1)
        A = interpolate(Expression((('1', 'x[0]'),
                                    ('2', 'x[1]')), degree=1), T)

        V = VectorFunctionSpace(mesh, 'CG', 1)
        v = interpolate(Expression(('x[0]+x[1]', '1'), degree=1), V)

        me = Eval(dot(v, transpose(A)))
        true = Expression(('x[1]+2*x[0]', '2*x[0]+3*x[1]'), degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)


    def test_sanity8(self):
        mesh = UnitSquareMesh(5, 5)
    
        V = VectorFunctionSpace(mesh, 'CG', 1)
        v0 = interpolate(Expression(('x[0]+x[1]', '1'), degree=1), V)
        v1 = interpolate(Expression(('1', 'x[0]'), degree=1), V)

        me = Eval(inner(v0, v1))
        true = Expression('x[1]+2*x[0]', degree=1)

        e = error(true, me)
        self.assertTrue(e < 1E-14)







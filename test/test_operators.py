from xcalc.interpreter import Eval
from xcalc.timeseries import TempSeries
from xcalc.operators import (Eigw, Eigv, Mean, RMS, STD, SlidingWindowFilter,
                             Minimum, Maximum)
from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) the operators'''
    def test_eigw(self):
        # FIxME: more advances

        A = np.array([[1, -2], [-3, 1]])

        mesh = UnitSquareMesh(10, 10)
        V = TensorFunctionSpace(mesh, 'DG', 0)
        f = interpolate(Constant(A), V)

        me = Eval(Eigw(f+f))  # Eval o Declared
        true = Constant(np.linalg.eigvals(A+A))

        self.assertTrue(error(true, me) < 1E-14)

    def test_eigv(self):        
        A = np.array([[1, -2], [-3, 1]])

        mesh = UnitSquareMesh(10, 10)
        V = TensorFunctionSpace(mesh, 'DG', 0)
        f = interpolate(Constant(A), V)
        # FIXME: 2*f leads to ComponentTensor which we don't handle well
        me = Eval(Eigv(f))
        true = Constant((np.linalg.eig(A)[1]).T)

        self.assertTrue(error(true, me) < 1E-14)

    def test_mean(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'CG', 1)
        f = Expression('(x[0]+x[1])*t', t=0, degree=1)
        
        ft_pairs = []
        for t in (0, 0.1, 0.4, 0.6, 2.0):
            f.t = t
            v = interpolate(f, V)
            ft_pairs.append((v, t))

        mean = Eval(Mean(TempSeries(ft_pairs)))  # Eval o Declared

        f.t = 1.0
        self.assertTrue(error(f, mean) < 1E-14)

    def test_rms(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, 'CG', 1)
        f = Expression('(x[0]+x[1])*t', t=0, degree=1)
        
        ft_pairs = []
        for t in np.linspace(0, 2, 80):
            f.t = t
            v = interpolate(f, V)
            ft_pairs.append((v, t))

        rms = Eval(RMS(TempSeries(ft_pairs)))

        f.t = sqrt(4/3.)
        # Due to quadrature error 
        self.assertTrue(error(f, rms) < 1E-4)

    def test_std(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, 'CG', 1)
        f = Expression('(x[0]+x[1])*t', t=0, degree=1)
        
        ft_pairs = []
        for t in np.linspace(0, 2, 80):
            f.t = t
            v = interpolate(f, V)
            ft_pairs.append((v, t))

        series = TempSeries(ft_pairs)

        std = Eval(STD(series))  # Efficiently in PETSc
        # From definition
        std_ = Eval(sqrt(Mean(series**2) - Mean(series)**2))

        self.assertTrue(error(std_, std) < 1E-14)

    def test_sliding_window(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, 'CG', 1)

        series = TempSeries([(interpolate(Constant(1), V), 0),
                             (interpolate(Constant(2), V), 1),
                             (interpolate(Constant(3), V), 2),
                             (interpolate(Constant(4), V), 3)])

        f_series = Eval(SlidingWindowFilter(Mean, 2, series**2))

        assert len(f_series) == 3
        assert f_series.times == (1, 2, 3)

        self.assertTrue(error(Constant(2.5), f_series.getitem(0)) < 1E-14)
        self.assertTrue(error(Constant(6.5), f_series.getitem(1)) < 1E-14)
        self.assertTrue(error(Constant(12.5), f_series.getitem(2)) < 1E-14)

    def test_minimum(self):
        A = np.array([[1, -2], [-3, 1]])

        mesh = UnitSquareMesh(10, 10)
        V = TensorFunctionSpace(mesh, 'DG', 0)
        f = interpolate(Constant(A), V)

        me = Eval(Minimum(Eigw(f+f)))  # Eval o Declared
        true = Constant(np.min(np.linalg.eigvals(A+A)))

        self.assertTrue(error(true, me) < 1E-14)

    def test_maximum(self):
        A = np.array([[1, -2], [-3, 1]])

        mesh = UnitSquareMesh(10, 10)
        V = TensorFunctionSpace(mesh, 'DG', 0)
        f = interpolate(Constant(A), V)
        
        me = Eval(Maximum(Eigw(f-3*f)))  # Eval o Declared
        true = Constant(np.max(np.linalg.eigvals(A-3*A)))

        self.assertTrue(error(true, me) < 1E-14)

    def test_maximum_fail(self):
        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, 'RT', 1)
        with self.assertRaises(AssertionError):
            Eval(Maximum(Function(V)))  # Don't know how to collapse this



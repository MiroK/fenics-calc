from xcalc.interpreter import Eval
from xcalc.timeseries import TempSeries
from xcalc.operators import Eigw, Eigv, Mean, RMS, SlidingWindowFilter, STD
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

        me = Eigw(f+f)
        true = Constant(np.linalg.eigvals(A+A))

        self.assertTrue(error(true, me) < 1E-14)

    def test_eigv(self):        
        A = np.array([[1, -2], [-3, 1]])

        mesh = UnitSquareMesh(10, 10)
        V = TensorFunctionSpace(mesh, 'DG', 0)
        f = interpolate(Constant(A), V)
        # FIXME: 2*f leads to ComponentTensor which we don't handle well
        me = Eigv(f)
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

        mean = Mean(TempSeries(ft_pairs))

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

        rms = RMS(TempSeries(ft_pairs))

        f.t = sqrt(4/3.)
        # Due to quadrature error 
        self.assertTrue(error(f, rms) < 1E-4)

    def test_sliding_window(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, 'CG', 1)

        series = TempSeries([(interpolate(Constant(1), V), 0),
                             (interpolate(Constant(2), V), 1),
                             (interpolate(Constant(3), V), 2),
                             (interpolate(Constant(4), V), 3)])

        f_series = SlidingWindowFilter(Mean, 2, series)

        assert len(f_series) == 3
        assert f_series.times == (1, 2, 3)

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

        std = STD(series)  # Efficiently in PETSc
        # From definition
        std_ = Eval(sqrt(Mean(series**2) - Mean(series)**2))

        self.assertTrue(error(std_, std) < 1E-14)



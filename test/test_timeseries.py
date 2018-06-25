from xcalc.interpreter import Eval
from xcalc.timeseries import TempSeries
from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) xcalc.timeseries'''
    def test_fail_on_times(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
        
        ft_pairs = ((Function(V), 0), (Function(V), -2))

        with self.assertRaises(AssertionError):
            TempSeries(ft_pairs)


    def test_fail_on_spaces(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
        W = FunctionSpace(mesh, 'CG', 1)
        
        ft_pairs = ((Function(V), 0), (Function(W), 1))

        with self.assertRaises(AssertionError):
            TempSeries(ft_pairs)

    def test_algebra_fail_different_times(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series0 = TempSeries(((Function(V), 0), (Function(V), 1)))
        series1 = TempSeries(((Function(V), 0), (Function(V), 2)))

        with self.assertRaises(AssertionError):
            Eval(series0 - series1)

    def test_algebra_fail_different_spaces(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
        W = FunctionSpace(mesh, 'CG', 1)
                
        series0 = TempSeries(((Function(V), 0), (Function(V), 1)))
        series1 = TempSeries(((Function(W), 0), (Function(W), 1)))

        with self.assertRaises(AssertionError):
            Eval(series0 - series1)
            
    def test_algebra(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series0 = TempSeries([(interpolate(Constant(1), V), 0),
                              (interpolate(Constant(2), V), 1)])

        series1 = TempSeries([(interpolate(Constant(2), V), 0),
                              (interpolate(Constant(3), V), 1)])

        series01 = Eval(series1 - series0)
        self.assertTrue(np.linalg.norm(series01.times - np.array([0, 1])) < 1E-14)

        # Now each should be 1
        for f in series01:
            self.assertTrue(error(Constant(1), f) < 1E-14)
            
    def test_vec_mag(self):
        mesh = UnitSquareMesh(2, 2)
        V = VectorFunctionSpace(mesh, 'CG', 1)
                
        series = TempSeries([(interpolate(Expression(('x[0]', '0'), degree=1), V), 0),
                              (interpolate(Expression(('0', 'x[1]'), degree=1), V), 1)])

        mag_series = Eval(sqrt(inner(series, series)))
        self.assertTrue(error(Expression('x[0]', degree=1), mag_series[0]) < 1E-14)
        self.assertTrue(error(Expression('x[1]', degree=1), mag_series[1]) < 1E-14)

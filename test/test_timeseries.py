from xcalc.interpreter import Eval
from xcalc.timeseries import TempSeries, stream, clip
from itertools import izip
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

    def test_steam(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series0 = TempSeries([(interpolate(Constant(1), V), 0),
                              (interpolate(Constant(2), V), 1)])

        v = Function(V)
        stream_series = stream(series0, v)
        # NOTE: it is crucial that this is lazy. With normal zip
        # v in all the pairse has the last value
        for vi, v in izip(series0, stream_series):
            self.assertTrue(error(vi, v) < 1E-14)

        for i, v in enumerate(stream_series):
            self.assertTrue(error(series0[i], v) < 1E-14)

    def test_clip(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series = TempSeries([(interpolate(Constant(1), V), 0),
                             (interpolate(Constant(2), V), 1),
                             (interpolate(Constant(3), V), 2),
                             (interpolate(Constant(4), V), 3)])

        clipped_series = clip(series, 0, 3)
        self.assertTrue(len(clipped_series)) == 2
        self.assertEqual(clipped_series.times, (1, 2))
        self.assertTrue(error(Constant(2), clipped_series[0]) < 1E-14)
        self.assertTrue(error(Constant(3), clipped_series[1]) < 1E-14)
        



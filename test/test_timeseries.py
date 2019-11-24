from xcalc.interpreter import Eval
from xcalc.timeseries import TempSeries, stream, clip, PVDTempSeries
import subprocess, time

from dolfin import *
import numpy as np
import unittest


def error(true, me):
    mesh = me.function_space().mesh()
    return sqrt(abs(assemble(inner(me - true, me - true)*dx(domain=mesh))))


class TestCases(unittest.TestCase):
    '''UnitTest for (some of) xcalc.timeseries'''
    @classmethod
    def setUpClass(cls):
        'called once, before any tests'
        subprocess.call(['mpirun -np 4 python3 setup_ppvtu.py'], shell=True, cwd='./test')
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        subprocess.call(['rm scalar* vector*'], shell=True, cwd='./test')
        time.sleep(3)
    
    def test_fail_on_times(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
        
        ft_pairs = ((Function(V), 0), (Function(V), -2))

        with self.assertRaises(AssertionError):
            TempSeries(ft_pairs)

    def test_fail_on_spaces(self):
        # Different element and degree in series
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)  
        W = FunctionSpace(mesh, 'CG', 1)
        
        ft_pairs = ((Function(V), 0), (Function(W), 1))

        with self.assertRaises(ValueError):
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
        self.assertTrue(error(Expression('x[0]', degree=1), mag_series.getitem(0)) < 1E-14)
        self.assertTrue(error(Expression('x[1]', degree=1), mag_series.getitem(1)) < 1E-14)

    def test_steam(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series0 = TempSeries([(interpolate(Constant(1), V), 0),
                              (interpolate(Constant(2), V), 1)])

        v = Function(V)
        stream_series = stream(2*series0, v)
        # NOTE: it is crucial that this is lazy. With normal zip
        # v in all the pairse has the last value
        for vi, v in zip(series0, stream_series):
            self.assertTrue(error(2*vi, v) < 1E-14)

        for i, v in enumerate(stream_series):
            self.assertTrue(error(2*series0.getitem(i), v) < 1E-14)

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
        self.assertTrue(error(Constant(2), clipped_series.getitem(0)) < 1E-14)
        self.assertTrue(error(Constant(3), clipped_series.getitem(1)) < 1E-14)

    def test_get(self):
        mesh = UnitSquareMesh(2, 2)
        V = VectorFunctionSpace(mesh, 'CG', 1)
                
        series = TempSeries([(interpolate(Expression(('x[0]', '0'), degree=1), V), 0),
                              (interpolate(Expression(('0', 'x[1]'), degree=1), V), 1)])

        mag_series = Eval(series[0])  # series of first componentsts
        self.assertTrue(error(Expression('x[0]', degree=1), mag_series.getitem(0)) < 1E-14)
        self.assertTrue(error(Expression('0', degree=1), mag_series.getitem(1)) < 1E-14)

        mag_series = Eval(series[1])  # series of secon componentsts
        self.assertTrue(error(Expression('0', degree=1), mag_series.getitem(0)) < 1E-14)
        self.assertTrue(error(Expression('x[1]', degree=1), mag_series.getitem(1)) < 1E-14)

        
    def test_algebra_harder(self):
        mesh = UnitSquareMesh(2, 2)
        V = FunctionSpace(mesh, 'DG', 0)
                
        series0 = TempSeries([(interpolate(Constant(2), V), 0),
                              (interpolate(Constant(3), V), 1)])

        series1 = TempSeries([(interpolate(Constant(4), V), 0),
                              (interpolate(Constant(5), V), 1)])

        series01 = Eval(series1**2 - 2*series0)
        self.assertTrue(np.linalg.norm(series01.times - np.array([0, 1])) < 1E-14)

        # Now each should be 1
        for f, true in zip(series01, (Constant(12), Constant(19))):
            self.assertTrue(error(true, f) < 1E-14)

        
    def test_pvtu_scalar(self):
        f = PVDTempSeries('./test/scalar.pvd', FiniteElement('Lagrange', triangle, 1))

        V = f.function_space()
        g = Expression('t*(x[0]+x[1])', degree=1, t=1)
        f0 = interpolate(g, V)

        for t, fi in enumerate(f, 1):
            g.t = t
            f0.assign(interpolate(g, V))

            self.assertTrue((fi.vector()-f0.vector()).norm('linf') < 1E-14)
            
    def test_pvtu_vector(self):
        f = PVDTempSeries('./test/vector.pvd', VectorElement('Lagrange', triangle, 1))

        V = f.function_space()
        g = Expression(('t*(x[0]+x[1])', 't'), degree=1, t=1)
        f0 = interpolate(g, V)

        for t, fi in enumerate(f, 1):
            g.t = t
            f0.assign(interpolate(g, V))

            self.assertTrue((fi.vector()-f0.vector()).norm('linf') < 1E-14)

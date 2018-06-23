# Some pseudo nodes (function constructors) that could be useful
# NOTE: unlike ufl nodes these are not lazy, i.e Eval immediately
from dolfin import Constant, interpolate, Function
from utils import numpy_op_foo
from interpreter import Eval
import numpy as np


def ConstantFunction(V, value):
    '''Constant over V'''
    if not isistance(V, Constant):
        return ConstantFunction(V, Constant(value))
    # Shape consistency
    assert V.ufl_element().value_shape() == value.ufl_shape
    
    return interpolate(value, V)


def Zero(V):
    '''Zero function over V'''
    return ConstantFunction(V, Constant(np.zeros(V.ufl_element().value_shape())))


def Eigw(expr):
    '''
    For a matrix-valued expression we make a vector-valued expression of eigenvalues
    '''
    n, m = expr.ufl_shape
    assert n == m, 'Square matrices only (or implement SVD)'

    f = Eval(expr)
    return numpy_op_foo(args=(f, ), op=np.linalg.eigvals, shape_res=(n, ))


def Eigv(expr):
    '''
    For a matrix-valued expression we make a matrix-valued expression where the rows
    have the eigenvector
    '''
    n, m = expr.ufl_shape
    assert n == m, 'Square matrices only (or implement SVD)'
    
    f = Eval(expr)
    return numpy_op_foo(args=(f, ),
                        op=lambda A: np.linalg.eig(A)[1].T,
                        shape_res=(n, m))

def Mean(series):
    '''A mean of the series is 1/(T - t0)\int_{t0}^{t1}f(t)dt'''
    # Apply simpsons rule
    mean = Function(series.V)
    x = mean.vector()
    # NOTE: for effiecency we stay away from Interpreter
    # Int
    dts = np.diff(series.times)
    for dt, (f0, f1) in zip(dts, zip(series.functions[:-1], series.functions[1:])):
        x.axpy(dt/2, f0.vector())  # (f0+f1)*dt/2
        x.axpy(dt/2, f1.vector())
    # Time interval scaling
    x /= dts.sum()

    return mean


def RMS(series):
    '''sqrt(1/(T - t0)\int_{t0}^{t1}f^2(t)dt'''
    # Again by applying simpson
    rms = Function(series.V)
    # NOTE: for efficiency we stay away from Interpreter and all is in PETSc layer
    x = as_backend_type(rms.vector()).vec()  # PETSc.Vec
    y = x.copy()  # Stores fi**2
    # Integrate
    dts = np.diff(series.times)
    f_vectors = [as_backend_type(f.vector()).vec() for f in series.functions]
    for dt, (f0, f1) in zip(dts, zip(f_vectors[:-1], f_vectors[1:])):
        y.pointwiseMult(f0, f0)  # y = f0**2
        x.axpy(dt/2, y)  # (f0**2+f1**2)*dt/2

        y.pointwiseMult(f1, f1)  # y = f1**2
        x.axpy(dt/2, y)
    # Time interval scaling
    x /= dts.sum()

    return rms
    
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, TensorFunctionSpace, Function, interpolate

    A = np.array([[1, -2], [-3, 1]])

    mesh = UnitSquareMesh(10, 10)
    V = TensorFunctionSpace(mesh, 'DG', 0)
    f = interpolate(Constant(A), V)

    print Eigw(f+f)(0.5, 0.5)    
    print np.linalg.eigvals(A+A)

    g = Eigv(f+f)
    print g(0.5, 0.5)    
    print np.linalg.eig(A+A)[1].T

# Some pseudo nodes (function constructors) that could be useful
# NOTE: unlike ufl nodes these are not lazy, i.e Eval immediately
from dolfin import Constant, interpolate, Function, as_backend_type
from collections import deque
from utils import numpy_op_foo
from interpreter import Eval
from timeseries import TempSeries
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
    series = Eval(series)  # Functions
    mean = Function(series.V)
    
    x = mean.vector()
    # NOTE: for effiecency we stay away from Interpreter
    # Int
    dts = np.diff(series.times)
    for dt, (f0, f1) in zip(dts, zip(series.nodes[:-1], series.nodes[1:])):
        x.axpy(dt/2., f0.vector())  # (f0+f1)*dt/2
        x.axpy(dt/2., f1.vector())
    # Time interval scaling
    x /= dts.sum()

    return mean


def RMS(series):
    '''sqrt(1/(T - t0)\int_{t0}^{t1}f^2(t)dt'''
    # Again by applying simpson
    series = Eval(series)
    rms = Function(series.V)
    # NOTE: for efficiency we stay away from Interpreter and all is in PETSc layer
    x = as_backend_type(rms.vector()).vec()  # PETSc.Vec
    y = x.copy()  # Stores fi**2
    # Integrate
    dts = np.diff(series.times)
    f_vectors = [as_backend_type(f.vector()).vec() for f in series.nodes]
    for dt, (f0, f1) in zip(dts, zip(f_vectors[:-1], f_vectors[1:])):
        y.pointwiseMult(f0, f0)  # y = f0**2
        x.axpy(dt/2., y)  # (f0**2+f1**2)*dt/2

        y.pointwiseMult(f1, f1)  # y = f1**2
        x.axpy(dt/2., y)
    # Time interval scaling
    x /= dts.sum()
    # sqrt
    x.sqrtabs()

    return rms


def STD(series):
    '''STD of series.'''
    # first, compute the mean
    mean = Mean(series)

    # get the square of the field of the mean
    mean_vector = mean.vector()
    mvs = as_backend_type(mean_vector).vec()
    # the mean squared, to be used for computing the RMS
    mvs.pointwiseMult(mvs, mvs)

    # now, compute the STD
    # for this, follow the example of RMS
    series = Eval(series)
    std = Function(series.V)
    # NOTE: for efficiency we stay away from Interpreter and all is in PETSc layer
    x = as_backend_type(std.vector()).vec()  # PETSc.Vec, stores the final output
    y = x.copy()  # Stores the current working field
    # Integrate
    dts = np.diff(series.times)
    f_vectors = [as_backend_type(f.vector()).vec() for f in series.nodes]
    for dt, (f0, f1) in zip(dts, zip(f_vectors[:-1], f_vectors[1:])):
        y.pointwiseMult(f0, f0)  # y = f0**2
        x.axpy(dt / 2., y)  # x += dt / 2 * y

        y.pointwiseMult(f1, f1)  # y = f1**2
        x.axpy(dt / 2., y)  # x += dt / 2 * y

        x.axpy(-dt, mvs)  # x += -dt * mvs  NOTE: no factor 2, as adding 2 dt / 2 to compensate
        
    # Time interval scaling
    x /= dts.sum()
    # sqrt
    x.sqrtabs()

    return std


def SlidingWindowFilter(Filter, width, series):
    '''
    Collapse a series into a different (shorter) series obtained by applying
    filter to the chunks of series of given width.
    '''
    assert width > 0
    t_buffer, f_buffer = deque(maxlen=width), deque(maxlen=width)

    times = series.times
    nodes = series.nodes

    filtered_ft_pairs = []
    for t, f in zip(times, nodes):
        t_buffer.append(t)
        f_buffer.append(f)
        # Once the deque is full it will 'overflow' from right so then
        # we have the right view to filter
        if len(f_buffer) == width:
            ff = Filter(TempSeries(zip(list(f_buffer), list(t_buffer))))
            tf = list(t_buffer)[width/2]  # Okay for odd

            filtered_ft_pairs.append((ff, tf))

    return TempSeries(filtered_ft_pairs)

# Some pseudo nodes (function constructors) that could be useful
# NOTE: unlike ufl nodes these are not lazy, i.e Eval immediately
from dolfin import Constant, interpolate, Function, as_backend_type
from collections import deque
from utils import numpy_op_foo, common_sub_element, make_space
from ufl.corealg.traversal import traverse_unique_terminals
import timeseries 
import interpreter
import numpy as np


class LazyNode(Function):
    '''Interac'''
    def __init__(self, V):
        Function.__init__(self, V)

    def evaluate(self):
        return Abstract

    @staticmethod
    def space_for(expr, shape=None):
        '''Function space where expr should be represented'''
        # Don't want to call eval here as it beats the goal of being lazy
        foos = filter(lambda f: isinstance(f, Function), traverse_unique_terminals(expr))
        _, = set(f.function_space().mesh().id() for f in foos)

        elm = common_sub_element([f.function_space() for f in foos])
        shape = expr.ufl_shape if shape is None else shape
        mesh = foos[0].function_space().mesh()
        
        return make_space(elm, shape, mesh)


class ConstantFunction(LazyNode):
    def __init__(self, V, value):
        self.value = Constant(value)
        # Don't allow declaring bullshit
        assert V.ufl_element().value_shape() == self.value.ufl_shape
        
        LazyNode.__init__(self, V)

    def evaluate(self):
        self.interpolate(self.value)
        return self

    
def Zero(V):
    '''Zero function over V'''
    return ConstantFunction(V, Constant(np.zeros(V.ufl_element().value_shape())))


class Eigw(LazyNode):
    '''
    For a matrix-valued expression we make a vector-valued expression of eigenvalues
    '''

    def __init__(self, expr):
        n, m = expr.ufl_shape
        assert n == m, 'Square matrices only (or implement SVD)'

        self.shape = (n, )
        self.expr = expr
        LazyNode.__init__(self, LazyNode.space_for(expr, self.shape))

    def evaluate(self):
        f = interpreter.Eval(self.expr)
        self.interpolate(numpy_op_foo(args=(f, ), op=np.linalg.eigvals, shape_res=self.shape))
        return self

    
class Eigv(LazyNode):
    '''
    For a matrix-valued expression we make a matrix-valued expression where the rows
    have the eigenvector
    '''
    def __init__(self, expr):
        n, m = expr.ufl_shape
        assert n == m, 'Square matrices only (or implement SVD)'

        self.shape = (n, m)
        self.expr = expr
        LazyNode.__init__(self, LazyNode.space_for(expr, self.shape))

    def evaluate(self):
        f = interpreter.Eval(self.expr)
        self.interpolate(numpy_op_foo(args=(f, ),
                                      op=lambda A: np.linalg.eig(A)[1].T,
                                      shape_res=self.shape))
        return self


class Mean(LazyNode):
    '''A mean of the series is 1/(T - t0)\int_{t0}^{t1}f(t)dt'''
    def __init__(self, expr):
        self.expr = expr
        LazyNode.__init__(self, LazyNode.space_for(expr))

    def evaluate(self):
        # Apply simpsons rule
        series = interpreter.Eval(self.expr)  # Functions

        mean = self
        x = mean.vector()
        x.zero()
        # NOTE: for effiecency we stay away from Interpreter
        # Int
        dts = np.diff(series.times)
        for dt, (f0, f1) in zip(dts, zip(series.nodes[:-1], series.nodes[1:])):
            x.axpy(dt/2., f0.vector())  # (f0+f1)*dt/2
            x.axpy(dt/2., f1.vector())
        # Time interval scaling
        x /= dts.sum()

        return self

    
class RMS(LazyNode):
    '''sqrt(1/(T - t0)\int_{t0}^{t1}f^2(t)dt'''
    def __init__(self, expr):
        self.expr = expr
        LazyNode.__init__(self, LazyNode.space_for(expr))

    def evaluate(self):
        # Again by applying simpson
        series = interpreter.Eval(self.expr)
        rms = self
        # NOTE: for efficiency we stay away from Interpreter and all is in PETSc layer
        x = as_backend_type(rms.vector()).vec()  # PETSc.Vec
        x.zeroEntries()
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

        return self


class STD(LazyNode):
    '''STD of series.'''
    def __init__(self, expr):
        self.expr = expr
        LazyNode.__init__(self, LazyNode.space_for(expr))

    def evaluate(self):
        # first, compute the mean
        series = interpreter.Eval(self.expr)
        mean = interpreter.Eval(Mean(series))

        # get the square of the field of the mean
        mean_vector = mean.vector()
        mvs = as_backend_type(mean_vector).vec()
        # the mean squared, to be used for computing the RMS
        mvs.pointwiseMult(mvs, mvs)

        # now, compute the STD
        # for this, follow the example of RMS
        std = self
        # NOTE: for efficiency we stay away from Interpreter and all is in PETSc layer
        x = as_backend_type(std.vector()).vec()  # PETSc.Vec, stores the final output
        x.zeroEntries()
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

        return self

    
def SlidingWindowFilter(Filter, width, series):
    '''
    Collapse a series into a different (shorter) series obtained by applying
    filter to the chunks of series of given width.
    '''
    assert width > 0
    t_buffer, f_buffer = deque(maxlen=width), deque(maxlen=width)

    series = interpreter.Eval(series)
    times = series.times
    nodes = series.nodes

    filtered_ft_pairs = []
    for t, f in zip(times, nodes):
        t_buffer.append(t)
        f_buffer.append(f)
        # Once the deque is full it will 'overflow' from right so then
        # we have the right view to filter
        if len(f_buffer) == width:
            ff = Filter(timeseries.TempSeries(zip(list(f_buffer), list(t_buffer))))
            tf = list(t_buffer)[width/2]  # Okay for odd

            filtered_ft_pairs.append((ff, tf))

    return timeseries.TempSeries(filtered_ft_pairs)

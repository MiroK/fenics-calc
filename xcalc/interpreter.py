from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import Function
import numpy as np
import ufl

from itertools import imap, repeat
import operator
import timeseries
from utils import *


def Eval(expr):
    '''
    This intepreter translates expr into a function object or a number. Expr is 
    defined via a subset of UFL language. Letting f, g be functions in V 
    Eval(op(f, g)) is a function in V with coefs given by (op(coefs(f), coef(g))).
    '''
    return Interpreter.eval(expr)


class Interpreter(object):
    '''
    This intepreter translates expr into a function object or a number. Expr is 
    defined via a subset of UFL language. Letting f, g be functions in V 
    Eval(op(f, g)) is a function in V with coefs given by (op(coefs(f), coef(g))).
    '''
    # Expression which when evaluated end up in the same space as the arguments
    # or require no reshaping of arraysbefore numpy is applied
    no_reshape_type = {
        ufl.algebra.Sum: np.add,
        ufl.algebra.Abs: np.abs,
        ufl.algebra.Division: np.divide,
        ufl.algebra.Product: np.multiply,
        ufl.algebra.Power: np.power,
        ufl.mathfunctions.Sin: np.sin,
        ufl.mathfunctions.Cos: np.cos,
        ufl.mathfunctions.Sqrt: np.sqrt,
        ufl.mathfunctions.Exp: np.exp, 
        ufl.mathfunctions.Ln: np.log,
        ufl.mathfunctions.Tan: np.tan,
        ufl.mathfunctions.Sinh: np.sinh,
        ufl.mathfunctions.Cosh: np.cosh,
        ufl.mathfunctions.Tanh: np.tanh,
        ufl.mathfunctions.Asin: np.arcsin,
        ufl.mathfunctions.Acos: np.arccos,
        ufl.mathfunctions.Atan: np.arctan,
        ufl.mathfunctions.Atan2: np.arctan2
    }

    # Expression which when evaluated end up in general in different space than 
    # the arguments/require manipulations before numpy is applied
    reshape_type = {
        ufl.tensoralgebra.Inverse: np.linalg.inv,
        ufl.tensoralgebra.Transposed: np.transpose,
        ufl.tensoralgebra.Sym: lambda A: 0.5*(A + A.T),
        ufl.tensoralgebra.Skew: lambda A: 0.5*(A - A.T),
        ufl.tensoralgebra.Deviatoric: lambda A: A - np.trace(A)*np.eye(len(A))*(1./len(A)),
        ufl.tensoralgebra.Cofactor: lambda A: np.linalg.det(A)*(np.linalg.inv(A)).T,
        ufl.tensoralgebra.Determinant: np.linalg.det,
        ufl.tensoralgebra.Trace: np.trace,
        ufl.tensoralgebra.Dot: np.dot,
        ufl.tensoralgebra.Cross: np.cross,
        ufl.tensoralgebra.Outer: np.outer,
        ufl.tensoralgebra.Inner: np.inner
    }
    
    # Other's where Eval works
    terminal_type = (Function, int, float, timeseries.TempSeries)
    value_type = (ufl.algebra.ScalarValue, ufl.algebra.IntValue)
    index_type = (ufl.indexed.Indexed, ufl.tensors.ComponentTensor)

    eval_types = reduce(operator.or_, (set(no_reshape_type.keys()),
                                       set(reshape_type.keys()),
                                       set(terminal_type),
                                       set(value_type),
                                       set(index_type)))

    @staticmethod
    def eval(expr):
        # Terminals/base cases (also TempSeries)
        if isinstance(expr, Interpreter.terminal_type): return expr

        if isinstance(expr, Interpreter.value_type): return expr.value()
        
        # Okay: now we have expr with arguments. If this expression involves 
        # times series then all the non number arguments should be compatible 
        # time series
        terminals = filter(lambda t: isinstance(t, Function), traverse_unique_terminals(expr))
        # Don't mix function and terminals
        series = filter(lambda t: isinstance(t, timeseries.TempSeries), terminals)

        assert len(series) == len(terminals) or len(series) == 0, map(len, (series, terminals))
        # For series, we apply op to functions and make new series
        if series:
            assert not isinstance(expr, Interpreter.index_type)
            return series_rule(expr)

        expr_type = type(expr)
        # Require reshaping and all args are functions
        if expr_type in Interpreter.reshape_type: 
            return numpy_reshaped(expr, op=Interpreter.reshape_type[expr_type])

        # Indexing [] is special as the second argument gives slicing
        if isinstance(expr, Interpreter.index_type): return indexed_rule(expr)

        # No reshaping neeed
        op = Interpreter.no_reshape_type[expr_type]  # Throw if we don't support this

        args = map(Interpreter.eval, expr.ufl_operands)
        # Manipulate coefs of arguments to get coefs of the expression
        coefs = map(coefs_of, args)
        V_coefs = op(*coefs)    
        # Make that function
        V = space_of(args)

        return make_function(V, V_coefs)


def numpy_reshaped(expr, op):
    '''Get the coefs by applying the numpy op to reshaped argument coefficients'''
    args = map(Interpreter.eval, expr.ufl_operands)

    # Exception to the rules are some ops with scalar args
    if isinstance(expr, (ufl.tensoralgebra.Inner, ufl.tensoralgebra.Dot)):
        if all(arg.ufl_shape == () for arg in args):
            return Interpreter.eval(args[0]*args[1])

    # Construct by numpy with op applied args of expr and reshaping as shape_res
    return numpy_op_foo(args, op=op, shape_res=expr.ufl_shape)


def indexed_rule(expr):
    '''Function representing f[index] so we end up with scalar'''
    shape_res = expr.ufl_shape
    # FIXME: A[:, 1] is a ComponentTensor. They are other ways to get that 
    # node and these would currently not be supported
    if isinstance(expr, ufl.tensors.ComponentTensor):
        expr = expr.ufl_operands[0]
        
    assert isinstance(expr, ufl.indexed.Indexed)
    f, index = expr.ufl_operands

    # What to index
    f = Interpreter.eval(f)
    # How to index 
    shape = f.ufl_shape
    indices = tuple(int(index) if isinstance(index, ufl.indexed.FixedIndex) else slice(l)
                    for l, index in zip(shape, index.indices()))
    # This could be implemented more efficiently (see earilier commits)
    # However, below is a more ideas which is that op is just a getitem
    op = lambda A, i=indices: A[i]
    
    return numpy_op_foo((f, ), op=op, shape_res=shape_res)


def series_rule(expr):
    '''Eval expression where the terminals are time series'''
    # Make first sure that the series are compatible in the sense
    # of having same f and time interval
    times = timeseries.common_interval(list(traverse_unique_terminals(expr)))
    assert len(times)

    series = map(Interpreter.eval, expr.ufl_operands)

    # We apply the op to functions in the series and construct a new one
    args = izip(*[s if isinstance(s, timeseries.TempSeries) else repeat(Interpreter.eval(s)) 
                  for s in series])

    functions = [Interpreter.eval(apply(type(expr), arg)) for arg in args]

    return timeseries.TempSeries(zip(functions, times))

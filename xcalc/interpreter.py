from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import (Function, VectorFunctionSpace, interpolate, Expression,
                    as_vector, Constant, as_matrix)
import numpy as np
import ufl
from itertools import imap, repeat, izip
import operator

import timeseries
import operators
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
        ufl.tensoralgebra.Inner: np.inner,
        ufl.tensors.ListTensor: lambda *x: np.hstack(x)
    }
    # FIXME: ListTensor(foo, indices=None) <= we have no support for indices
    
    # Other's where Eval works
    terminal_type = (Function, int, float) 
    value_type = (ufl.algebra.ScalarValue, ufl.algebra.IntValue)  
    index_type = (ufl.indexed.Indexed, )
    compose_type = (ufl.tensors.ComponentTensor, )

    @staticmethod
    def eval(expr):
        # Guys with their own logic for collapsing into functions
        # Okay we combine 2 design patters, LazyNodes do it themselves
        # series rely on the interpreter
        if isinstance(expr, operators.LazyNode):
            return expr.evaluate()
       
        # For series we eval each node and make a series of functions
        # NOTE: intersept here because TempSeries is a terminal type
        if isinstance(expr, timeseries.TempSeries):
            return timeseries.TempSeries(zip(map(Interpreter.eval, expr), expr.times))

        # Terminals/base cases (also TempSeries) -> identity
        if isinstance(expr, Interpreter.terminal_type): return expr

        # To number
        if isinstance(expr, Interpreter.value_type): return expr.value()

        # To number
        if isinstance(expr, Constant): return float(expr)

        # Recast spatial coordinate as CG1 functions
        if isinstance(expr, ufl.geometry.SpatialCoordinate):
            mesh = expr.ufl_domain().ufl_cargo()
            r = Expression(('x[0]', 'x[1]', 'x[2]')[:mesh.geometry().dim()], degree=1)
            return interpolate(r, VectorFunctionSpace(mesh, 'CG', 1))
        
        # Okay: now we have expr with arguments. If this expression involves 
        # times series then all the non number arguments should be compatible 
        # time series
        terminals = filter(lambda t: isinstance(t, Function), traverse_unique_terminals(expr))
        # Don't mix function and terminals
        series = filter(lambda t: isinstance(t, timeseries.TempSeries), terminals)

        assert len(series) == len(terminals) or len(series) == 0, map(type, terminals)
        # For series, we apply op to functions and make new series
        if series:
            return series_rule(expr)

        expr_type = type(expr)
        # Require reshaping and all args are functions
        if expr_type in Interpreter.reshape_type: 
            return numpy_reshaped(expr, op=Interpreter.reshape_type[expr_type])

        # Define tensor by componenents
        if isinstance(expr, Interpreter.compose_type):
            return component_tensor_rule(expr)

        # A indexed by FixedIndex or Index
        if isinstance(expr, Interpreter.index_type):
            return indexed_rule(expr)

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
    foos = filter(lambda f: isinstance(f, Function), traverse_unique_terminals(expr))
    # Make first sure that the series are compatible in the sense of having same time 
    # interval
    times = timeseries.common_interval(foos)
    assert len(times)

    # Compatibility of spaces
    common_sub_element([f.function_space() for f in foos])

    # The idea now is to propagate the expression by which I mean that
    # we grow the expr using nodes in the series
    def unpack(expr):
       '''expr -> iterable of expr'''
       return (apply(type(expr), sum(args, ())) for args in expand(expr.ufl_operands))

    def expand(operands):
       iterators = []
       for o in operands:
          if isinstance(o, timeseries.TempSeries):
             iterators.append(((f, ) for f in o))
          # Nonseries terminal
          elif not o.ufl_operands:
             iterators.append(((f, ) for f in repeat(o)))
          # An expression
          else:
             iterators.append(((f, ) for f in unpack(o)))
         
       return izip(*iterators)

    nodes = unpack(expr)
    # A series of new nodes -> series of functions
    return Interpreter.eval(timeseries.TempSeries(zip(nodes, times)))


def component_tensor_rule(expr):
   '''Tensors whose components are given by computation of some sort.'''
   f, free_indices = expr.ufl_operands
   # Want to build vectors or matrices
   assert len(free_indices) == 1 or len(free_indices) == 2

   # Simple rules where the eval node is obtained just by substitution
   if not isinstance(f, ufl.indexsum.IndexSum):
      # Vector from 2*Constant((1, 2)) 
      if len(free_indices) == 1:
         index = free_indices[0]
         f = tuple(replace(f, index, FixedIndex(i)) for i in range(expr.ufl_shape[0]))
         
         return Interpreter.eval(as_vector(f))
        
      # Matrix from 2*Costant(((1, 2), (3, 4)))
      if len(free_indices) == 2:
         mat = []
         for i in range(expr.ufl_shape[0]):
            f_i = replace(f, free_indices[0], FixedIndex(i))

            row = []
            for j in range(expr.ufl_shape[1]):
               row.append(replace(f_i, free_indices[1], FixedIndex(j)))
            mat.append(row)
         return Interpreter.eval(as_matrix(mat))
         
   # The idea now is to to build the expression which represents the sum
   # needed to compute the component, i.e. explicit transformation of the
   # IndexSum node. Computing with scalars this way is not very efficient ->
   # FIXME: drop to numpy?
   assert isinstance(f, ufl.indexsum.IndexSum)

   summand, sum_indices = f.ufl_operands
   assert len(sum_indices) == 1  # FIXME: is this necessary

   # Be explicit about the sum - have free indices left to be fill
   # in by that component
   sum_expr = sum(replace(summand, sum_indices[0], FixedIndex(j))
                  for j in range(f.dimension()))

   # Now build the components
   if len(free_indices) == 1:
      # Sub for the free_i
      expr = as_vector(tuple(replace(sum_expr, free_indices[0], FixedIndex(i))
                             for i in range(f.ufl_index_dimensions[0])))

      return Interpreter.eval(expr)

   mat = []
   for i in range(f.ufl_index_dimensions[0]):
      # Sub i
      sub_i = replace(sum_expr, free_indices[0], FixedIndex(i))
      
      row = []
      for j in range(f.ufl_index_dimensions[1]):
         # Sub j
         row.append(replace(sub_i, free_indices[1], FixedIndex(j)))
      mat.append(row)
   expr = as_matrix(mat)
      
   return Interpreter.eval(expr)

from dolfin import Function
import numpy as np
import ufl

from itertools import imap
from utils import *


def Eval(expr):
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

    # Terminals/base cases
    if isinstance(expr, (Function, int, float)):
        return expr

    if isinstance(expr, (ufl.algebra.ScalarValue, ufl.algebra.IntValue)):
        return expr.value()
        
    expr_type = type(expr)
    # Require reshaping and all args are functions
    if expr_type in reshape_type:
        return numpy_reshaped(expr, op=reshape_type[expr_type])

    # Indexing [] is special as the second argument gives slicing
    if isinstance(expr, ufl.indexed.Indexed):
        return indexed_rule(expr)

    # No reshaping neeed
    args = map(Eval, expr.ufl_operands)

    op = no_reshape_type[expr_type]
    # Manipulate coefs of arguments to get coefs of the expression
    coefs = map(coefs_of, args)
    V_coefs = op(*coefs)    
    # Make that function
    V = space_of(args)
    return make_function(V, V_coefs)


def numpy_reshaped(expr, op):
    '''Get the coefs by applying the numpy op to reshaped argument coefficients'''
    args = map(Eval, expr.ufl_operands)

    # Exception to the rules are some ops with scalar args
    if isinstance(expr, (ufl.tensoralgebra.Inner, ufl.tensoralgebra.Dot)):
        if all(a.ufl_shape == () for arg in args):
            return Eval(args[0]*args[1])

    # Construct by numpy with op applied args of expr and reshaping as shape_res
    return numpy_op_foo(args, op=op, shape_res=expr.ufl_shape)


def indexed_rule(expr):
    '''Function representing f[index] so we end up with scalar'''
    f, index = expr.ufl_operands
    # Don't allow for slices
    assert all(isinstance(i, ufl.indexed.FixedIndex) for i in index.indices())
    # What to index
    f = Eval(f)
    V = f.function_space()
    # Make sure that this is tensor product space
    elm_indexed = common_sub_element((V, ))
    # We want to flat the index to be used with dofmap extracting
    index = flat_index(map(int, index.indices()), f.ufl_shape)
    # Get the values
    coefs = coefs_of(f)
    coefs_indexed = coefs[V.sub(index).dofmap().dofs()]
    # Shape of the value must be scalar
    V_indexed = make_space(elm_indexed, (), V.mesh())
    return make_function(V_indexed, coefs_indexed)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)

    f = Expression('x[0]', degree=1)
    g = Expression('x[1]', degree=1)
    a = 1
    b = 2

    u = interpolate(f, V)
    v = interpolate(g, V)

    expr = a*u + b*v

    me = Eval(expr)
        
    true = Expression('a*f+b*g', f=f, g=g, a=a, b=b, degree=1)
            
    print assemble(inner(me-true, me-true)*dx)

    # ---- 

    mesh = UnitSquareMesh(5, 5)
    T = TensorFunctionSpace(mesh, 'DG', 0)

    u = interpolate(Constant(((0, 1), (2, 3))), T)
    expr = sym(u) + skew(u)
    true = u

    me = Eval(expr)
    print assemble(inner(me-true, me-true)*dx)

    # -----

    V = FunctionSpace(mesh, 'DG', 0)

    expr = tr(sym(u) + skew(u))

    me = Eval(expr)
    true = interpolate(Constant(3), V)
    print assemble(inner(me-true, me-true)*dx(domain=mesh))
    
    # -----

    expr = sym(u)[1, 1]
    me = Eval(expr)
    true = interpolate(Constant(3), V)
    print assemble(inner(me-true, me-true)*dx(domain=mesh))


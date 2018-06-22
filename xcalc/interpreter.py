from dolfin import Function, FunctionSpace
import numpy as np
import ufl



UFL_2_NUMPY = {ufl.algebra.Sum: np.add,
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
               ufl.mathfunctions.Atan2: np.arctan2}

def Eval(expr, numpy_rules=UFL_2_NUMPY):
    '''
    This intepreter translates expr into a function object or a number. Expr is 
    defined via a subset of UFL language. Letting f, g be functions in V 
    Eval(op(f, g)) is a function in V with coefs given by (op(coefs(f), coef(g))).
    '''
    # Terminals/base cases
    if isinstance(expr, (Function, int, float)):
        return expr

    if isinstance(expr, (ufl.algebra.ScalarValue, ufl.algebra.IntValue)):
        return expr.value()

    # Inv, Transpose, Sym, Skew, Dev, Cross maps matrices to matrices
    if isinstance(expr, (ufl.tensoralgebra.Inverse,
                         ufl.tensoralgebra.Transposed,
                         ufl.tensoralgebra.Sym,
                         ufl.tensoralgebra.Skew,
                         ufl.tensoralgebra.Deviatoric,
                         ufl.tensoralgebra.Cofactor)):
        return mat_to_mat(expr)

    # Tr and Det nodes map matrix to scalar. We enable this only if the 
    # space if V x V ....
    if isinstance(expr, (ufl.tensoralgebra.Determinant, ufl.tensoralgebra.Trace)):
        return mat_to_scalar(expr)

    # Cross, Dot, Outer map tensors to tensor/scalar
    if isinstance(expr, (ufl.tensoralgebra.Dot,
                         ufl.tensoralgebra.Cross,
                         ufl.tensoralgebra.Outer,
                         ufl.tensoralgebra.Inner)):
        return tensor_to_tensor(expr)

    # NOTE: for now we assume that all the expression arguments are either 
    # numbers or functions in the same function space. We can get the coefficients
    # by straight forward manipulations of the coefficient arrays
    args = map(Eval, expr.ufl_operands)
    # Things from here on must be possible via numpy
    op = numpy_rules[type(expr)]
    # Manipulate coefs of arguments to get coefs of the expression
    coefs = map(coefs_of, args)
    V_coefs = op(*coefs)    
    # Make that function
    V = space_of(args)
    return make_function(V, V_coefs)


def make_function(V, coefs):
    '''A function in V with coefs'''
    f = Function(V)
    f.vector().set_local(coefs)
    # FIXME: parallel

    return f


def coefs_of(f):
    '''Extract coefficients of f'''
    # We arrive here either with a function or a number
    if isinstance(f, Function):
        return f.vector().get_local()
    assert isinstance(f, (int, float)), (f, type(f))

    return f


def space_of(foos):
    '''Extract the function space for representing foos'''
    # We arrive here either with a function or a number
    elm, mesh = None, None
    for f in filter(lambda x: isinstance(x, Function), foos):
        elm_ = f.function_space().ufl_element()
        mesh_ = f.function_space().mesh()

        if elm is None:
            elm = elm_
            mesh = mesh_
        else:
            assert elm_ == elm and mesh.id() == mesh_.id()

    return FunctionSpace(mesh, elm)


def mat_to_mat(expr):
    '''Single argument nodes that represent transformation of matrix to matrix'''
    A = Eval(expr.ufl_operands[0]) 
    V = A.function_space()
    coefs = coefs_of(A)
    # New values are obtained by mapping op numpy over the right values
    shape = A.ufl_shape
    n, m = shape
    indices = np.column_stack([V.sub(comp).dofmap().dofs()
                               for comp in range(V.num_sub_spaces())])
    # NOTE: doing this by Eval(0.5*(A-tr(A))) would require handling 
    # fewer ops here but we'd need ConstantFunction
    ops = {ufl.tensoralgebra.Inverse: np.linalg.inv,
           ufl.tensoralgebra.Transposed: np.transpose,
           ufl.tensoralgebra.Sym: lambda A: 0.5*(A + A.T),
           ufl.tensoralgebra.Skew: lambda A: 0.5*(A - A.T),
           ufl.tensoralgebra.Deviatoric: lambda A, n=n: A - np.trace(A)*np.eye(n)*(1./n),
           ufl.tensoralgebra.Cofactor: lambda A, n=n: np.linalg.det(A)*(np.linalg.inv(A)).T}
    op = ops[type(expr)]

    for row in indices:
        coefs[row] = op(coefs[row].reshape(shape)).flatten()
    return make_function(V, coefs)


def mat_to_scalar(expr):
    '''Single argument nodes that represent transformation of matrix to scalar'''
    A = Eval(expr.ufl_operands[0]) 
    V = A.function_space()
    # Is this a V x V x ... x V space
    sub_elm,  = set(V.ufl_element().sub_elements())
    # If so proceed to compute the values

    values = coefs_of(A)
    # New values are obtained by mapping op numpy over the right values
    shape = A.ufl_shape
    n, m = shape
    indices = np.column_stack([V.sub(comp).dofmap().dofs()
                               for comp in range(V.num_sub_spaces())])

    op = {ufl.tensoralgebra.Determinant: np.linalg.det,
          ufl.tensoralgebra.Trace: np.trace}[type(expr)]
    
    # Getting values of the scalar function
    coefs = np.fromiter((op(values[row].reshape(shape)) for row in indices), dtype=float)
    V = FunctionSpace(V.mesh(), sub_elm)
    
    return make_function(V, coefs)


def tensor_to_tensor(expr):
    '''Two argument nodes that represent transformation between tensors'''
    A, B = map(Eval, expr.ufl_operands)
    # Do we have V x V spaces?
    subA = 
    subB =


    V = A.function_space()
    coefs = coefs_of(A)
    # New values are obtained by mapping op numpy over the right values
    shape = A.ufl_shape
    n, m = shape
    indices = np.column_stack([V.sub(comp).dofmap().dofs()
                               for comp in range(V.num_sub_spaces())])


    # NOTE: doing this by Eval(0.5*(A-tr(A))) would require handling 
    # fewer ops here but we'd need ConstantFunction
    ops = {ufl.tensoralgebra.Inverse: np.linalg.inv,
           ufl.tensoralgebra.Transposed: np.transpose,
           ufl.tensoralgebra.Sym: lambda A: 0.5*(A + A.T),
           ufl.tensoralgebra.Skew: lambda A: 0.5*(A - A.T),
           ufl.tensoralgebra.Deviatoric: lambda A, n=n: A - np.trace(A)*np.eye(n)*(1./n),
           ufl.tensoralgebra.Cofactor: lambda A, n=n: np.linalg.det(A)*(np.linalg.inv(A)).T}
    op = ops[type(expr)]

    for row in indices:
        coefs[row] = op(coefs[row].reshape(shape)).flatten()
    return make_function(V, coefs)


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
    

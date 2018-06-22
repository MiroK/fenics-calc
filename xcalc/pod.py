from interpreter import Eval
from dolfin import inner, dx, assemble, Function, sqrt
from types import FunctionType
import numpy as np


def linear_combination(coefs, foos):
    '''Construct a new function as a linear combinations sum_i coefs[i]*foos[i]'''
    assert all(isinstance(f, Function) for f in foos)
    assert len(coefs) == len(foos)
    assert len(foos)

    f = sum(ci*fi for ci, fi in zip(coefs, foos))
    return Eval(f)


def normalize(f, ip):
    '''Normalize f to be ip(f, f) = 1'''
    return Eval(f/sqrt(ip(f, f)))


def nargs(f): 
    '''Argument count of a function'''
    return f.__code__.co_argcount


def pod(functions, ip='l2'):
    '''
    Proper orthogonal decomposition

    Let there be a collection of Functions and an inner product. POD constructs 
    basis of the space spanned by functions according to the eigendecomposition 
    of the inner product matrix.
    '''
    # Sanity of inputs
    assert all(isinstance(f, Function) for f in functions)

    # Predefined inner products
    if ip == 'l2':
        ip = lambda u, v: u.vector().inner(v.vector())

    elif ip == 'L2':
        ip = lambda u, v: assemble(inner(u, v)*dx)

    elif ip == 'H1':
        ip = lambda u, v: assemble(inner(u, v)*dx + inner(grad(u), grad(v))*dx)

    assert isinstance(ip, FunctionType) and nargs(ip) == 2

    # Build the (symmetric) matrix of the inner products
    n = len(functions)

    A = np.zeros((n, n))
    for i, fi in enumerate(functions):
        A[i, i] = ip(fi, fi)
        for j, fj in enumerate(functions[i+1:], i+1):
            A[i, j] = ip(fi, fj)
            A[j, i] = A[j, i]

    eigw, eigv = np.linalg.eigh(A)
    # Make eigv have rows as vectors
    eigv = eigv.T
    # New basis function are linear combinations with weights given by eigv[i]
    pod_basis = [linear_combination(c, functions) for c in eigv]
    # Normalize to unity in the inner product
    pod_basis = [normalize(f, ip) for f in pod_basis]

    return eigw, pod_basis

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, Expression, FunctionSpace, interpolate, File
    # Build a monomial basis for x, y, x**2, xy, y**2, ...

    deg = 4

    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, 'CG', 1)
    x = interpolate(Expression('x[0]', degree=1), V)
    y = interpolate(Expression('x[1]', degree=1), V)

    basis = []
    for i in range(deg):
        for j in range(deg):
            basis.append(Eval((x**i)*(y**j)))

    # NOTE: skipping 1 bacause Eval of it is not a Function
    energy, pod_basis = pod(basis[1:])

    out = File('pod_test.pvd')
    for i, f in enumerate(pod_basis):
        f.rename('f', '0')
        out << (f, float(i))

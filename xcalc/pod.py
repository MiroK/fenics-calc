from dolfin import inner, dx, assemble, Function, sqrt
from types import FunctionType
import numpy as np


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
            value = ip(fi, fj)
            A[i, j] = A[j, i] = value

    eigw, eigv = np.linalg.eigh(A)
    # Make eigv have rows as vectors
    eigw = np.sqrt(eigw)
    eigv = eigv.T
    # New basis function are linear combinations with weights given by eigv[i]
    pod_basis = [linear_combination(c, functions) for c in eigv]
    # Normalize to unity in the inner product
    pod_basis = [normalize(f, ip) for f in pod_basis]

    return eigw, pod_basis


def linear_combination(coefs, foos):
    '''Construct a new function as a linear combinations sum_i coefs[i]*foos[i]'''
    assert all(isinstance(f, Function) for f in foos)
    assert len(coefs) == len(foos)
    assert len(foos)

    # For the reasons of speed we do this in C (no Eval)
    f = Function(foos[0].function_space())  # Zero
    F = f.vector()
    for ci, fi in zip(coefs, foos):
        F.axpy(ci, fi.vector())

    return f


def normalize(f, ip):
    '''Normalize f to be ip(f, f) = 1'''
    f.vector()[:] *= 1./sqrt(ip(f, f))
    return f


def nargs(f): 
    '''Argument count of a function'''
    return f.__code__.co_argcount

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, Expression, FunctionSpace, interpolate, File
    from dolfin import XDMFFile
    from interpreter import Eval
    # Build a monomial basis for x, y, x**2, xy, y**2, ...

    deg = 4

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, 'CG', 3)
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


    with XDMFFile(mesh.mpi_comm(), 'pod_test.xdmf') as out:
        for i, f in enumerate(pod_basis):
            f.rename('f', '0')
            out.write(f, float(i))

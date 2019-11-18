from dolfin import Function
from types import FunctionType
import numpy as np


def pod(functions, ip=lambda u, v: u.vector().inner(v.vector()), modal_analysis=[]):
    '''
    Proper orthogonal decomposition

    Let there be a collection of Functions and an inner product. POD constructs 
    basis of the space spanned by functions according to the eigendecomposition 
    of the inner product matrix. Let that basis be {phi_i}. Each fj of functions 
    can be decomposed in the basis as fj = c^{j}_i phi_i. If j is associated with 
    time then c^{all j}_i give a temporal evolution of the coef for mode i.
    With `modal_analysis` a matrix C is returned which corresponds to 
    coef of i-th modes.
    '''
    # Sanity of inputs
    assert all(isinstance(f, Function) for f in functions)

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
    # NOTE: the matrix should normally be pos def but round of ...
    eigw = np.abs(eigw)
    # Make eigv have rows as vectors
    eigv = eigv.T
    # Reverse so that largest modes come first
    eigw = eigw[::-1]
    eigv = eigv[::-1]
    
    # New basis function are linear combinations with weights given by eigv[i]
    pod_basis = [linear_combination(c, functions, np.sqrt(a)) for c, a in zip(eigv, eigw)]

    if not modal_analysis:
        return eigw, pod_basis

    C = np.array([[ip(pod_basis[i], fj) for fj in functions] for i in modal_analysis])

    return eigw, pod_basis, C
        

def linear_combination(coefs, foos, scale=1):
    '''Construct a new function as a linear combinations (1./scale)*sum_i coefs[i]*foos[i]'''
    assert all(isinstance(f, Function) for f in foos)
    assert len(coefs) == len(foos)
    assert len(foos)

    # For the reasons of speed we do this in C (no Eval)
    f = Function(foos[0].function_space())  # Zero
    F = f.vector()
    for ci, fi in zip(coefs, foos):
        F.axpy(ci, fi.vector())
    F /= scale

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
    from dolfin import XDMFFile, inner, grad, dx, assemble
    from .interpreter import Eval
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

    ip = lambda u, v: u.vector().inner(v.vector())
    #ip = lambda u, v: assemble(inner(u, v)*dx)
    #ip = lambda u, v: assemble(inner(u, v)*dx + inner(grad(u), grad(v))*dx)

    # NOTE: skipping 1 bacause Eval of it is not a Function
    energy, pod_basis = pod(basis[1:], ip=ip)

    out = File('pod_test.pvd')
    for i, f in enumerate(pod_basis):
        f.rename('f', '0')
        out << (f, float(i))


    with XDMFFile(mesh.mpi_comm(), 'pod_test.xdmf') as out:
        for i, f in enumerate(pod_basis):
            f.rename('f', '0')
            out.write(f, float(i))

    for fi in pod_basis:
        for fj in pod_basis:
            print(ip(fi, fj))
        print()

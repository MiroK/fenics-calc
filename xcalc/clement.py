from dolfin import *
from mpi4py import MPI as piMPI
import ufl


def clement_interpolate(expr):
    '''
    Here, the Clement interpolant is a CG_1 function over 
    mesh constructed in two steps (See Braess' Finite element book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (i.e wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj.
    '''
    # Analyze expr and raise if invalid
    terminals = _analyze_expr(expr)
    # Analyze shape and raise if expr cannot be represented
    _analyze_shape(expr.ufl_shape)
    shape = expr.ufl_shape
    # Extract mesh from expr operands and raise if it is not unique or missing
    mesh = _extract_mesh(terminals)
    # Compute things for constructing Q
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(Q)
    # Forms for L2 means [rhs]
    # Scalar, Vectors, Tensors are built from components
    # Translate expression into forms for individual components
    if len(shape) == 0:
        forms = [inner(expr, q)*dx]
    elif len(shape) == 1:
        forms = [inner(expr[i], q)*dx for i in range(shape[0])]
    else:
        forms = [inner(expr[i, j], q)*dx for i in range(shape[0]) for j in range(shape[1])]

    # Build averaging or summation operator for computing the interpolant
    # from L2 averaged components.
    V = FunctionSpace(mesh, 'CG', 1)
    volumes = assemble(inner(Constant(1), q)*dx)
    # Ideally we compute the averaging operator, then the interpolant is
    # simply A*component. I have not implemented this for backends other 
    # than PETSc. 
    is_petsc = parameters['linear_algebra_backend'] == 'PETSc'
    assert is_petsc
    
    A = _construct_averaging_operator(V, volumes)
        
    # L2 means of comps to indiv. cells
    means = list(map(assemble, forms))

    # The interpolant (scalar, vector, tensor) is build from components
    components = []
    for mean in means:
         # Scalar
         component = Function(V)
         A.mult(mean, component.vector()) 
         components.append(component)
        
    # Finalize the interpolant
    # Scalar has same space as component
    if len(shape) == 0: 
         uh = components.pop()
         uh.vector().apply('insert')

    # We can precompute maps for assigning the components
    if len(shape) == 1:
        W = VectorFunctionSpace(mesh, 'CG', 1, dim=shape[0])
    else:
        W = TensorFunctionSpace(mesh, 'CG', 1, shape=shape)
    assigner = FunctionAssigner(W, [V]*len(forms))

    uh = Function(W)
    assigner.assign(uh, components)
    
    uh.vector().apply('insert')

    return uh

# Workers--

def _analyze_expr(expr):
    '''
    A valid expr for Clement interpolation is defined only in terms of pointwise
    operations on finite element functions.
    '''
    # Cannot interpolate expression with Arguments + things which are not well
    # defined at vertex
    black_listed = (ufl.Argument, ufl.MaxCellEdgeLength, ufl.MaxFacetEdgeLength,
                    ufl.MinCellEdgeLength, ufl.MinFacetEdgeLength,
                    ufl.FacetArea, ufl.FacetNormal, 
                    ufl.CellNormal, ufl.CellVolume)

    # Elliminate forms
    if isinstance(expr, ufl.Form): raise ValueError('Expression is a form')
    # Elliminate expressions build from Trial/Test functions, FacetNormals 
    terminals = [t for t in ufl.corealg.traversal.traverse_unique_terminals(expr)]
    if any(isinstance(t, black_listed) for t in terminals):
        raise ValueError('Invalid expression (e.g. has Arguments as operand)')
    # At this point the expression is valid
    return terminals


def _analyze_shape(shape):
    '''
    The shape of expr that UFL can build is arbitrary but we only support
    scalar, rank-1 and rank-2(square) tensors.
    '''
    is_valid = len(shape) < 3 and (shape[0] == shape[1] if len(shape) == 2 else True)
    if not is_valid:
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')


def _extract_mesh(terminals):
    '''Get the common mesh of operands that make the expression.'''
    pairs = []
    for t in terminals:
        try: 
            mesh = t.function_space().mesh()
            pairs.append((mesh.id(), mesh))
        except AttributeError:
            try:
                mesh = t.ufl_domain().ufl_cargo()
                pairs.append((mesh.id(), mesh))
            except AttributeError:
                pass
    ids = set(id_ for id_, _ in pairs)
    # Unique mesh
    if len(ids) == 1: return pairs.pop()[1]
    # Mesh of Nones of multiple
    raise ValueError('Failed to extract mesh: Operands with no or different meshes')


def _construct_summation_operator(V):
    '''
    Summation matrix has the following properties: It is a map from DG0 to CG1.
    It has the same sparsity pattern as the mass matrix and in each row the nonzero
    entries are 1. Finally let v \in DG0 then (A*v)_i is the sum of entries of v
    that live on the support of i-th basis function of CG1.
    '''
    mesh = V.mesh()
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TrialFunction(Q)
    v = TestFunction(V)
    tdim = mesh.topology().dim()
    K = CellVolume(mesh)
    dX = dx(metadata={'form_compiler_parameters': {'quadrature_degree': 1,
                                                   'quadrature_scheme': 'vertex'}})
    # This is a nice trick which uses properties of the vertex quadrature to get
    # only ones as nonzero entries.
    # NOTE: Its is designed spec for CG1. In particular does not work CG2 etc so
    # for such spaces a difference construction is required, e.g. rewrite nnz
    # entries of mass matric V, Q to 1. That said CG2 is the highest order where
    # clement interpolation makes sense. With higher ordered the dofs that are
    # interior to cell (or if there are multiple dofs par facet interior) are
    # assigned the same value.
    A = assemble((1./K)*Constant(tdim+1)*inner(v, q)*dX)

    return A


def _construct_averaging_operator(V, c):
    '''
    If b is the vectors of L^2 means of some u on the mesh, v is the vector
    of cell volumes and A is the summation oparotr then x=(Ab)/(Ac) are the
    coefficient of Clement interpolant of u in V. Here we construct an operator
    B such that x = Bb.
    '''
    assert parameters['linear_algebra_backend'] == 'PETSc'
   
    A = _construct_summation_operator(V)

    Ac = Function(V).vector()
    A.mult(c, Ac)
    # 1/Ac
    Ac = as_backend_type(Ac).vec()
    Ac.reciprocal()     
    # Scale rows
    mat = as_backend_type(A).mat()
    mat.diagonalScale(L=Ac)

    return A

from dolfin import Function, FunctionSpace, VectorElement, TensorElement
from itertools import imap, izip


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


def numpy_op_indices(V, shape):
    '''Iterator over dofs of V in a logical way'''
    # next(numpy_of_indices(V)) gets indices for accessing coef of function in V
    # in a way that after reshaping the values can be used by numpy
    nsubs = V.num_sub_spaces()
    # Get will give us e.g matrix to go with det to set the value of det
    if nsubs:
        assert len(shape)
        indices = imap(list, izip(*[iter(V.sub(comp).dofmap().dofs()) for comp in range(nsubs)]))
    else:
        assert not len(shape)
        indices = iter(V.dofmap().dofs())

    return indices


def common_sub_element(spaces):
    '''V for space which are tensor products of V otherwise fail'''
    V = None
    for space in spaces:
        if not space.num_sub_spaces():
            V_ = space.ufl_element()
        # V x V x V ... => V
        else:
            V_, = set(space.ufl_element().sub_elements())
        # Unset or agrees
        assert V is None or V == V_
        V = V_
    # All is well
    return V


def make_space(V, shape, mesh):
    '''Tensor product space of right shape'''
    if not shape:
        elm = V
    elif len(shape) == 1:
        elm = VectorElement(V, len(shape))
    elif len(shape) == 2:
        elm = TensorElement(V, shape)
    else:
        raise ValueError('No spaces for tensor of rank 3 and higher')

    return FunctionSpace(mesh, elm)


def numpy_op_foo(args, op, shape_res):
    '''Construct function with shape_res ufl_shape by applying op to args'''
    # Do we have V x V x ... spaces?
    sub_elm = common_sub_element([space_of((arg, )) for arg in args])
    
    get_args = []
    # Construct iterators for accesing the coef values of arguments in the 
    # right way be used with numpy op
    for arg in args:
        arg_coefs = coefs_of(arg)

        V = arg.function_space()
        shape = arg.ufl_shape

        # How to access coefficients by indices 
        indices = numpy_op_indices(V, shape)

        # Get values for op by reshaping
        if shape:
            get = imap(lambda i, c=arg_coefs, s=shape: c[i].reshape(s), indices)
        else:
            get = imap(lambda i, c=arg_coefs: c[i], indices)

        get_args.append(get)
    # Now all the arguments can be iterated to gether by
    args = izip(*get_args)

    # Construct the result space
    V_res = make_space(sub_elm, shape_res, V.mesh())
    # How to reshape the result and assign
    if shape_res:
        dofs = imap(list, numpy_op_indices(V_res, shape_res))
        reshape = lambda x: x.flatten()
    else:
        dofs = numpy_op_indices(V_res, shape_res)
        reshape = lambda x: x
        
    # Fill coefs of the result expression
    coefs_res = Function(V_res).vector().get_local()
    for dof, dof_args in izip(dofs, args):
        coefs_res[dof] = reshape(op(*dof_args))
    # NOTE: make_function so that there is only one place (hopefully)
    # where parallelism needs to be addressed
    return make_function(V_res, coefs_res)



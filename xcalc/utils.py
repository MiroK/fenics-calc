from dolfin import Function, FunctionSpace, VectorElement, TensorElement
from itertools import imap, izip, dropwhile, ifilterfalse, ifilter

from ufl.indexed import Index, FixedIndex, MultiIndex
from ufl.core.terminal import Terminal


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


def numpy_op_indices(V):
    '''Iterator over dofs of V in a logical way'''
    # next(numpy_of_indices(V)) gets indices for accessing coef of function in V
    # in a way that after reshaping the values can be used by numpy
    nsubs = V.num_sub_spaces()
    # Get will give us e.g matrix to go with det to set the value of det
    if nsubs:
        indices = imap(list, izip(*[iter(V.sub(comp).dofmap().dofs()) for comp in range(nsubs)]))
    else:
        indices = iter(V.dofmap().dofs())

    return indices


def common_sub_element(spaces):
    '''V for space which are tensor products of V otherwise fail'''
    V = None
    for space in spaces:
        V_ = component_element(space)        
        # Unset or agrees
        assert V is None or V == V_
        V = V_
    # All is well
    return V


def component_element(elm):
    '''If the space/FE has a structure V x V ... x V find V'''
    # Type convert
    if isinstance(elm, FunctionSpace):
        return component_element(elm.ufl_element())
    # Single component
    if not elm.sub_elements():
        return elm
    # V x V x V ... => V
    V_, = set(elm.sub_elements())

    return V_


def shape_representation(shape, elm):
    '''How to reshape expression of shape represented in FE space with elm'''
    celm = component_element(elm)
    # Scalar is a base
    if not celm.value_shape():
        return shape
    # Can't represent vector with matrix space
    eshape = celm.value_shape()
    assert len(shape) >= len(eshape)

    # Vec with vec requires no reshaping
    if shape == eshape:
        return ()
    # Compatibility
    assert shape[-len(eshape):] ==  eshape
    # So (2, 2) with (2, ) is (2, )
    return shape[:len(eshape)]


def make_space(V, shape, mesh):
    '''Tensor product space of right shape'''
    finite_elements = [lambda x, shape: x,
                       lambda x, shape: VectorElement(x, dim=shape[0]),
                       lambda x, shape: TensorElement(x, shape=shape)]
    # FEM upscales; cant upscale larger
    assert len(shape) - len(V.value_shape()) >= 0
    # No tensor
    assert len(shape) <= 2

    fe_glue = finite_elements[len(shape) - len(V.value_shape())]
    elm = fe_glue(V, shape)
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
        shape = shape_representation(arg.ufl_shape, V.ufl_element())

        # How to access coefficients by indices
        indices = numpy_op_indices(V)

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
    if shape_representation(shape_res, V_res.ufl_element()):
        dofs = imap(list, numpy_op_indices(V_res))
        reshape = lambda x: x.flatten()
    else:
        dofs = numpy_op_indices(V_res)
        reshape = lambda x: x
        
    # Fill coefs of the result expression
    coefs_res = Function(V_res).vector().get_local()
    for dof, dof_args in izip(dofs, args):
        coefs_res[dof] = reshape(op(*dof_args))
    # NOTE: make_function so that there is only one place (hopefully)
    # where parallelism needs to be addressed
    return make_function(V_res, coefs_res)

# Utils for series

def find_first(things, predicate):
    '''Index of first item in container which satisfies the predicate'''
    return next(dropwhile(lambda i, s=things: not predicate(s[i]), range(len(things))))


def find_last(things, predicate):
    '''Counting things backward the index of the first item satisfying the predcate'''
    return -find_first(list(reversed(things)), predicate)-1


def clip_index(array, first, last):
    '''Every item x in array[clip_index(...)] satisfied first < x < last'''
    assert first < last
    f = find_first(array, lambda x, f=first: x > f)
    l = find_last(array, lambda x, l=last: x < l) + 1
    
    return slice(f, l)

# UFL utils for substitution of indices

def is_index(expr):
    return isinstance(expr, (Index, FixedIndex))


def traverse_indices(expr):
    '''Traverse the UFL expression (drilling into indices)'''
    if expr.ufl_operands:
        for op in expr.ufl_operands:
            for e in ifilter(is_index, traverse_indices(op)):
                yield e
    # Multiindex has no operands but we want the indices
    if isinstance(expr, MultiIndex):
        for i in expr.indices():
            yield i

    
def matches(expr, target):
    '''Compare two indices for equalty'''
    return expr == target

                                                              
def contains(expr, target):
    '''Is the target index contained in the expression?'''
    # A tarminal target either agrees or is one of the expr terminals
    if is_index(expr):
        return expr == target
    else:
        return any(matches(target, t) for t in traverse_indices(expr))

    
def replace(expr, arg, replacement):
    '''A new expression where argument in the expression is the replacement'''
    # Do nothing if no way to substitute, i.e. return original
    if not contains(expr, arg):
        return expr
    # Identical 
    if matches(expr, arg):
        return replacement
    # Reconstruct the node with the substituted argument
    if expr.ufl_operands:
        return type(expr)(*[replace(op, arg, replacement) for op in expr.ufl_operands])
    # This has to be MultiIndex
    return MultiIndex(tuple(replace(op, arg, replacement) for op in expr.indices()))

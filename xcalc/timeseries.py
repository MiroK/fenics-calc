import xml.etree.ElementTree as ET
from .function_read import (read_h5_function, read_vtu_function,
                           read_h5_mesh, read_vtu_mesh)
from dolfin import (Function, XDMFFile, HDF5File, FunctionSpace,
                    VectorFunctionSpace, TensorFunctionSpace) #, warning)
from ufl.corealg.traversal import traverse_unique_terminals
from .utils import space_of, clip_index
from . import interpreter
import numpy as np
import itertools
import os


class TempSeries(Function):
    '''
    Collection of snapshots that are when Eval are functions in same 
    space V. That is, series are lazy in general. UFL nodes are supported 
    over series with logic op([s], [t]) = [op(s, t)].
    '''
    def __init__(self, ft_pairs):
        # NOTE: this is derived from Function just to allow nice
        # interplay with the interpreter. If there were space time 
        # elements then we could have eval f(t, x) support
        nodes, times = list(zip(*ft_pairs))
        
        # Checks some necessaru conditions for compatibility of nodes in the series
        assert check_nodes(nodes)
        # Optimistically take the function space
        V = interpreter.Eval(next(iter(nodes))).function_space()
        # Time interval check
        dt = np.diff(times)
        assert (dt > 0).all()
        
        self.nodes = nodes
        self.times = times
        self.V = V

        Function.__init__(self, V)

    def __iter__(self):
        '''Iterate nodes in the series'''
        # op(series) = series(op(functions))
        for f in self.nodes: yield f 

    def __len__(self):
        return len(self.nodes)

    def getitem(self, index):
        '''Access elements of the time series'''
        if isinstance(index, int):
            return self.nodes[index]
        else:
            return TempSeries(list(zip(self.nodes[index], self.times[index])))

        
def stream(series, f):
    '''Pipe series through Function f'''
    series = interpreter.Eval(series)
    assert series.V.ufl_element() == f.function_space().ufl_element()
    assert series.V.mesh().id() == f.function_space().mesh().id()

    for f_ in series:  # Get your own iterator
        f.vector().set_local(f_.vector().get_local())
        yield f


def clip(series, t0, t1):
    '''A view of the series with times such that t0 < times < t1'''
    index = clip_index(series.times, t0, t1)
    nodes = series.nodes[index]
    times = series.times[index]

    return TempSeries(list(zip(nodes, times)))

        
def common_interval(series):
    '''Series are compatible if they have same intervals'''
    series = [s for s in series if isinstance(s, TempSeries)]

    interval = []
    for s in series:
        interval_ = np.array(s.times)
        assert not len(interval) or np.linalg.norm(interval - interval_) < 1E-14

        interval = interval_
    return interval


def check_nodes(series):
    '''
    Nodes in the series are said to be compatible here iff

    1) they are over the same mesh
    2) they have the same base element
    3) they have the same shape
    '''
    shape, = set(f.ufl_shape for f in series)

    terminal_functions = lambda s=series: (
        filter(lambda f: isinstance(f, Function),
                          itertools.chain(*list(map(traverse_unique_terminals, s))))
    )
    # Base element                                                                                 
    family, = set(f.ufl_element().family() for f in terminal_functions())
    degree, = set(f.ufl_element().degree() for f in terminal_functions())

    # Mesh
    mesh_id, = set(f.function_space().mesh().id() for f in terminal_functions())
    
    return True


def get_P1_space(V):
    '''Get the Lagrange CG1 space corresponding to V'''
    # This is how in essence FEniCS 2017.2.0 dumps data, i.e. there is
    # no support for higher order spaces
    assert V.ufl_element().family() != 'Discontinuous Lagrange'  # Cell data needed
    
    mesh = V.mesh()
    elm = V.ufl_element()
    if elm.value_shape() == ():
        return FunctionSpace(mesh, 'CG', 1)
    
    if len(elm.value_shape()) == 1:
        return VectorFunctionSpace(mesh, 'CG', 1)

    return TensorFunctionSpace(mesh, 'CG', 1)


def PVDTempSeries(path, V=None, first=0, last=None):
    '''
    Read in the temp series of functions in V from PVD file. If V is not 
    a function space then a finite element has to be provided for constructing
    the space on the recovered mesh.
    '''
    _, ext = os.path.splitext(path)
    assert ext == '.pvd'

    tree = ET.parse(path)
    collection = list(tree.getroot())[0]
    path = os.path.dirname(os.path.abspath(path))
    # Read in paths/timestamps for VTUs. NOTE: as thus is supposed to be serial 
    # assert part 0
    vtus, times = [], []
    for dataset in collection:
        assert dataset.attrib['part'] == '0'
        
        vtus.append(os.path.join(path, dataset.attrib['file']))
        times.append(float(dataset.attrib['timestep']))
    
    vtus, times = vtus[slice(first, last, None)], times[slice(first, last, None)]
    # path.vtu -> function. But vertex values!!!!
    
    if not isinstance(V, FunctionSpace):
        warning('Setting up P1 space on the recovered mesh')

        cell_type = V.cell()  # Dangerously assuming this is a UFL element
        mesh = read_vtu_mesh(vtus[0], cell_type)
        
        V = FunctionSpace(mesh, V)
    V = get_P1_space(V)

    functions = read_vtu_function(vtus, V)
    ft_pairs = list(zip(functions, times))

    return TempSeries(ft_pairs)


def XDMFTempSeries(path, V, first=0, last=None):
    '''
    Read in the temp series of functions in V from XDMF file. If V is not 
    a function space then a finite element has to be provided for constructing
    the space on the recovered mesh.
    '''
    # NOTE: in 2017.2.0 fenics only stores vertex values so CG1 functions
    # is what we go for
    _, ext = os.path.splitext(path)
    assert ext == '.xdmf'

    tree = ET.parse(path)
    domain = list(tree.getroot())[0]
    grid = list(domain)[0]

    times = []  # Only collect time stamps so that we access in right order
    h5_file = ''  # Consistency of piece as VisualisationVector ...
    for item in grid:
        _, __, time, attrib = list(item)
        time = time.attrib['Value']
        times.append(time)

        piece = list(attrib)[0]
        h5_file_, fdata = piece.text.split(':/')

        assert not h5_file or h5_file == h5_file_
        h5_file = h5_file_
        
    times = times[slice(first, last, None)]
    # We read visualization vector from this
    h5_file = os.path.join(os.path.dirname(os.path.abspath(path)), h5_file)

    if not isinstance(V, FunctionSpace):
        warning('Setting up P1 space on the recovered mesh')

        cell_type = V.cell()  # Dangerously assuming this is a UFL element
        mesh = read_h5_mesh(h5_file, cell_type)
        
        V = FunctionSpace(mesh, V)
    V = get_P1_space(V)

    functions = read_h5_function(h5_file, times, V)
    
    ft_pairs = list(zip(functions, list(map(float, times))))

    return TempSeries(ft_pairs)

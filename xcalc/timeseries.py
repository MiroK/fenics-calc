import xml.etree.ElementTree as ET
from function_read import read_h5_function
from dolfin import (Function, XDMFFile, HDF5File, FunctionSpace,
                    VectorFunctionSpace, TensorFunctionSpace)
from utils import space_of
import numpy as np
import os


class TempSeries(Function):
    '''Collection of snapshots that are function in same V'''
    def __init__(self, ft_pairs):
        # NOTE: this is derived from Function just to allow nice
        # interplay with the interpreter. If there were space time 
        # elements then we could have eval f(t, x) support
        functions, times = list(zip(*ft_pairs))
        
        # Check that every f has same f
        V = space_of(functions)
        # Time interval check
        dt = np.diff(times)
        assert (dt > 0).all()
        
        self.functions = functions
        self.times = times
        self.V = V

        Function.__init__(self, V)

    def __iter__(self):
        # op(series) = series(op(functions))
        for f in self.functions: yield f 

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index):
        # NOTE: by having this [] is access to functions not a new time series
        # of the components
        if isinstance(index, int):
            return self.functions[index]
        else:
            return TempSeries(zip(self.functions[index], self.times[index]))


def common_interval(series):
    '''Series are compatible if they over same V and cover same interval'''
    series = filter(lambda s: isinstance(s, TempSeries), series)

    interval, V = [], None
    for s in series:
        V_ = s.V
        assert V is None or (V.mesh().id() == V_.mesh().id()
                             and 
                             V.ufl_element() == V_.ufl_element())

        interval_ = s.times
        assert not interval or np.linalg.norm(interval - interval_) < 1E-14

        V = V_
        interval = interval_
    return interval


def PVDTempSeries(path, V, first=0, last=None):
    '''
    Read in the temp series of functions from PVD file
    '''
    # NOTE: vtu does NOT save higher than linear polynomials
    from vtk_io import read_vtu_function

    _, ext = os.path.splitext(path)
    assert ext == '.pvd'

    tree = ET.parse(path)
    collection = list(tree.getroot())[0]
    # Read in paths/timestamps for VTUs. NOTE: as thus is supposed to be serial 
    # assert part 0
    vtus, times = [], []
    for dataset in collection:
        assert dataset.attrib['part'] == '0'
        vtus.append((dataset.attrib['file'], float(dataset.attrib['timestep'])))
    
    vtus = vtus[slice(first, last, None)]
    # path.vtu -> function. But vertex values!!!!
    ft_pairs = [(read_vtu_function(path, V), t) for path, t in vtus]

    return TempSeries(ft_pairs)


def XDMFTempSeries(path, V, first=0, last=None):
    '''Read in the temp series of functions from XDMF file'''
    # NOTE: in 2017.2.0 fenics only stores vertex values so CG1 functions
    # is what we go for
    mesh = V.mesh()
    elm = V.ufl_element()
    if elm.value_shape() == ():
        V = FunctionSpace(mesh, 'CG', 1)
    elif len(elm.value_shape()) == 1:
        V = VectorFunctionSpace(mesh, 'CG', 1)
    else:
        V = TensorFunctionSpace(mesh, 'CG', 1)

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
    h5_file = os.path.join(os.getcwd(), h5_file)
    functions = read_h5_function(h5_file, times, V)
    
    ft_pairs = zip(functions, map(float, times))

    return TempSeries(ft_pairs)
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from dolfin import *

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    a = Function(V)
    b = Function(V)

    c = TempSeries([(a, 0), (b, 1)])
    print len(c)

    for f in c:
        print f

    print type(c[0:1])

    # --- Check scalar
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, 'CG', 1)
    f0 = interpolate(Expression('x[0]', degree=1), V)
    f1 = interpolate(Expression('x[0]', degree=1), V)

    with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
        f0.rename('f', '0')
        out.write(f0, 0.)

        f1.rename('f', '0')
        out.write(f1, 1.)

    # PVDTempSeries('pod_test.pvd', V)
    series = XDMFTempSeries('xdmf_test.xdmf', V)
    print assemble(inner(f0 - series[0], f0 - series[0])*dx(domain=mesh))
    print assemble(inner(f1 - series[1], f1 - series[1])*dx(domain=mesh))

    # --- Check vector
    V = VectorFunctionSpace(mesh, 'CG', 1)
    f0 = interpolate(Expression(('x[0]', 'x[1]'), degree=1), V)
    f1 = interpolate(Expression(('2*x[0]', '-3*x[1]'), degree=1), V)

    with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
        f0.rename('f', '0')
        out.write(f0, 0.)

        f1.rename('f', '0')
        out.write(f1, 1.)

    series = XDMFTempSeries('xdmf_test.xdmf', V)
    print assemble(inner(f0 - series[0], f0 - series[0])*dx(domain=mesh))
    print assemble(inner(f1 - series[1], f1 - series[1])*dx(domain=mesh))
    exit()
    
    # --- Check tensor
    V = TensorFunctionSpace(mesh, 'CG', 1)
    f0 = interpolate(Expression((('x[0]', 'x[1]'), ('x[0]', 'x[1]')),degree=1),
                     V)
    f1 = interpolate(Expression((('2*x[0]', '-3*x[1]'), ('2*x[0]', '-3*x[1]')), degree=1),
                     V)

    print f0.vector().local_size()
    with XDMFFile(mesh.mpi_comm(), 'xdmf_test.xdmf') as out:
        f0.rename('f', '0')
        out.write(f0, 0.)

        f1.rename('f', '0')
        out.write(f1, 1.)


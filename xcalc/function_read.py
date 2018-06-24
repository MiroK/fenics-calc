# This is the most fragile component of the package so be advised that 
# these ARE NOT GENERAL PURPOSE READEDERS


from dolfin import Function, dof_to_vertex_map
from mpi4py import MPI
import numpy as np
import h5py

assert MPI.COMM_WORLD.size == 1, 'No parallel for your own good'


def read_h5_function(h5_file, times, V):
    '''
    Read in function in V from h5_file:/VisualisationVector/times
    '''
    gdim = V.mesh().geometry().dim()
    assert gdim > 1
    
    if isinstance(times, str): times = [times]

    # H5 stores 3d vectors and 3d tensor so we need to chop the data
    # also reorder as in 2017.2.0 only(?) vertex values are dumped
    if V.ufl_element().value_shape() == ():
        dof2v = dof_to_vertex_map(V)
        reorder = lambda a: a[dof2v]
    else:
        Vi = V.sub(0).collapse()
        dof2v = dof_to_vertex_map(Vi)
        # WARNING: below there are assumption on component ordering
        # Vector
        if len(V.ufl_element().value_shape()) == 1:
            # Ellim Z for vectors in 2d
            keep = [0, 1] if gdim == 2 else range(gdim)

            reorder = lambda a, keep=keep, dof2f=dof2v:(
                np.column_stack([row[dof2v] for row in (a[:, keep]).T]).flatten()
            )
        # And tensor
        elif len(V.ufl_element().value_shape()) == 2:
            # Ellim Z
            keep = [0, 1, 3, 4] if gdim == 2 else range(gdim**2)
            
            reorder = lambda a, keep=keep, dof2f=dof2v:(
                np.column_hstack([row[dof2v] for row in (a[:, keep]).T]).flatten()
            )
            
    # Read the functions
    comm = V.mesh().mpi_comm()

    functions = []
    with h5py.File(h5_file, 'r') as h5:
        group = h5.get('VisualisationVector')
        for key in times:
            f = Function(V)  # What to fill

            data = group[key].value
            f.vector().set_local(reorder(data))
            
            functions.append(f)

    return functions

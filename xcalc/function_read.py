# This is the most fragile component of the package so be advised that 
# these ARE NOT GENERAL PURPOSE READEDERS


from dolfin import (Function, dof_to_vertex_map, Mesh, MeshEditor, SubsetIterator,
                    DomainBoundary, MeshFunction)
import xml.etree.ElementTree as ET
from itertools import dropwhile
from mpi4py import MPI
import numpy as np
import os

try:
    import h5py
except ImportError:
    print('H5Py missing')


assert MPI.COMM_WORLD.size == 1, 'No parallel (for your own good)'


def data_reordering(V):
    '''Reshaping/reordering data read from files'''
    # HDF5/VTK store 3d vectors and 3d tensor so we need to chop the data
    # also reorder as in 2017.2.0 only(?) vertex values are dumped
    if V.ufl_element().value_shape() == ():
        dof2v = dof_to_vertex_map(V)
        reorder = lambda a: a[dof2v]

        return reorder
    
    Vi = V.sub(0).collapse()
    dof2v = dof_to_vertex_map(Vi)
    gdim = V.mesh().geometry().dim()
    # WARNING: below there are assumption on component ordering
    # Vector
    if len(V.ufl_element().value_shape()) == 1:
        # Ellim Z for vectors in 2d
        keep = [0, 1] if gdim == 2 else list(range(gdim))

        reorder = lambda a, keep=keep, dof2f=dof2v:(

            np.column_stack([row[dof2v] for row in (a[:, keep]).T]).flatten()
        )

        return reorder
    
    # And tensor
    if len(V.ufl_element().value_shape()) == 2:
        # Ellim Z
        keep = [0, 1, 3, 4] if gdim == 2 else list(range(gdim**2))
            
        reorder = lambda a, keep=keep, dof2f=dof2v:(
            np.column_stack([row[dof2v] for row in (a[:, keep]).T]).flatten()
        )

        return reorder

    
def read_vtu_function(vtus, V, aux_mesh=None):
    '''Read in functions in V from VTUs files'''
    # NOTE: this would much easier with (py)vtk but that is not part of
    # the FEniCS stack so ...
    gdim = V.mesh().geometry().dim()
    assert gdim > 1
    if isinstance(vtus, str): vtus = [vtus]

    reorder = data_reordering(V)

    npoints, ncells = V.mesh().num_vertices(), V.mesh().num_cells()
    functions = []
    for vtu in vtus:
        f = Function(V)

        if len(vtu) == 1:
            data = read_vtu_point_data(vtu, npoints, ncells)
        else:
            assert aux_mesh is not None

            # Now we fill with vertex ordering
            data = None
            for v in vtu:
                rank = vtu_rank(v)
                d = read_vtu_point_data(v)

                if data is None:
                    if d.ndim == 2:
                        data = np.zeros((npoints, d.shape[1]))
                    else:
                        data = np.zeros(npoints)
                data[aux_mesh.leafs[rank]] = d

        f.vector().set_local(reorder(data))
        functions.append(f)
    return functions


def read_vtu_point_data(vtus, nvertices=None, ncells=None):
    '''PointData element of ASCII VTU file'''
    if isinstance(vtus, str): vtus = (vtus, )

    vtu, = vtus
    tree = ET.parse(vtu)
    root = tree.getroot()
    grid = next(iter(root))
    piece = next(iter(grid))

    # Check consistency of mesh (somewhat)
    assert nvertices is None or nvertices == int(piece.attrib['NumberOfPoints'])
    assert ncells is None or ncells == int(piece.attrib['NumberOfCells'])
    # Throw StopIteration
    point_data_elm = next(dropwhile(lambda x: x.tag != 'PointData', piece))
    data = next(iter(point_data_elm))

    ncomps = int(data.attrib.get('NumberOfComponents', 0))
    values = np.array(list(map(float, list(filter(bool, data.text.split(' '))))))
    # Reshape for reorder (so it is same as H5File
    if ncomps:
        values = values.reshape((-1, ncomps))
    return values


def read_h5_function(h5_file, times, V):
    '''
    Read in function in V from h5_file:/VisualisationVector/times
    '''
    gdim = V.mesh().geometry().dim()
    assert gdim > 1
    
    if isinstance(times, str): times = [times]

    reorder = data_reordering(V)

    functions = []
    # Read the functions
    with h5py.File(h5_file, 'r') as h5:
        group = h5.get('VisualisationVector')
        for key in times:
            f = Function(V)  # What to fill

            data = group[key].value
            f.vector().set_local(reorder(data))
            
            functions.append(f)
    return functions


def read_h5_mesh(path, cell_type):
    '''Read in mesh from function stored in H5 file'''
    # Is there a better way? (via HDF5File)
    h5 = h5py.File(path, 'r')

    mesh_group = h5['Mesh']['0']['mesh']
    vertices = mesh_group['geometry'].value
    cells = mesh_group['topology'].value

    return make_mesh(vertices, cells, cell_type)


def union_mesh(meshes, tol=1E-12):
    '''Glue together meshes into a big one.'''
    assert meshes

    num_meshes = len(meshes)
    # Nothing to do
    if num_meshes == 1:
        return meshes[0]
    # Recurse
    if num_meshes > 2:
        return union_mesh([union_mesh(meshes[:num_meshes//2+1]),
                           union_mesh(meshes[num_meshes//2+1:])])

    gdim, = set(m.geometry().dim() for m in meshes)
    tdim, = set(m.topology().dim() for m in meshes)
    cell_type, = set(m.ufl_cell() for m in meshes)
    
    fdim = tdim-1
    bdries = [MeshFunction('size_t', m, fdim, 0) for m in meshes]
    [DomainBoundary().mark(bdry, 1) for bdry in bdries]
    # We are after boundary vertices of both; NOTE that the assumption
    # here is that the meshes share only the boundary vertices
    [m.init(fdim) for m in meshes]
    [m.init(fdim, 0) for m in meshes]

    bdry_vertices0, bdry_vertices1 = map(list, (set(np.concatenate([f.entities(0) for f in SubsetIterator(bdry, 1)]))
                                                for bdry in bdries))
    
    x0, x1 = [m.coordinates() for m in meshes]

    x1 = x1[bdry_vertices1]
    shared_vertices = {}
    while bdry_vertices0:
        i = bdry_vertices0.pop()
        x = x0[i]
        # Try to match it
        dist = np.linalg.norm(x1 - x, 2, axis=1)
        imin = np.argmin(dist)
        if dist[imin] < tol:
            shared_vertices[bdry_vertices1[imin]] = i
            x1 = np.delete(x1, imin, axis=0)
            del bdry_vertices1[imin]

    mesh0, mesh1 = meshes
    # We make 0 the master - it adds all its vertices
    # The other on add all but those that are not shared
    unshared = list(set(range(mesh1.num_vertices())) - set(shared_vertices.keys()))
    
    merge_x = mesh0.coordinates()
    offset = len(merge_x)
    # Vertices of the merged mesh
    merge_x = np.row_stack([merge_x, mesh1.coordinates()[unshared]])
    # Mapping for cells from meshes
    lg1 = {k: v for v, k in enumerate(unshared, offset)}
    lg1.update(shared_vertices)
    # Collapse to list
    _, lg1 = zip(*sorted(lg1.items(), key=lambda v: v[0]))
    lg1 = np.array(lg1)
    
    mapped_cells = np.fromiter((lg1[v] for v in np.concatenate(mesh1.cells())),
                               dtype='uintp').reshape((mesh1.num_cells(), -1))
    merged_cells = np.row_stack([mesh0.cells(), mapped_cells])

    merged_mesh = make_mesh(merge_x, merged_cells, cell_type)

    lg0 = np.arange(mesh0.num_vertices())
    # Mapping from leafs
    if not hasattr(mesh0, 'leafs'):
        merged_mesh.leafs = [(mesh0.id(), lg0)]
    else:
        merged_mesh.leafs = mesh0.leafs
        
    if not hasattr(mesh1, 'leafs'):
        merged_mesh.leafs.append([mesh1.id(), lg1])
    else:
        for id_, map_ in mesh1.leafs:
            merged_mesh.leafs.append((id_, lg1[map_]))
                
    return merged_mesh


def vtu_rank(f):
    '''Exprect rank from vtu file name'''
    # We look for something like _
    root, ext = os.path.splitext(os.path.basename(f))
    assert ext == '.vtu'

    try:
        idx = root.index('_p')
    except ValueError:
        return 0

    idx = idx + len('_p')
    rank = int(root[idx:root.index('_', idx)])

    return rank


def read_vtu_mesh(paths, cell_type):
    '''Read in mesh from function stored in vtu file'''
    if isinstance(paths, str): paths = (paths, )
    # Will need to glue together
    if len(paths) > 1:
        meshes = [read_vtu_mesh(p, cell_type) for p in paths]
        mesh = union_mesh(meshes)

        # We now want to build mapping from rank corresponding to pieces
        # (meshes) to local2global vtx map
        ranks = list(map(vtu_rank, paths))
        ids = [m.id() for m in meshes]
        
        leaf_map = {}  # <- this one

        leafs = dict(mesh.leafs)
        while leafs:
            id_, map_ = leafs.popitem()

            i = ids.index(id_)
            rank = ranks[i]
            leaf_map[rank] = map_

            ids.remove(id_)
            ranks.remove(rank)
            
        # Processor -> mapping
        mesh.leafs = leaf_map

        return mesh
    # Base Case
    path, = paths
    
    tree = ET.parse(path)
    root = tree.getroot()
    grid = next(iter(root))
    piece = next(iter(grid))

    points, cells, _ = list(piece)
    
    # Parse points
    point_data = next(iter(points))
    # Always 3d gdim with this file format
    gdim = cell_type.geometric_dimension()
    point_data = np.array(list(map(float, list(filter(bool, point_data.text.split(' '))))))
    point_data = point_data.reshape((-1, 3))[:, :gdim]

    # Parse cells
    cell_data = next(iter(cells))
    cell_data = np.array(list(map(int, list(filter(bool, cell_data.text.split(' '))))))
    cell_data = cell_data.reshape((-1, cell_type.num_vertices()))

    return make_mesh(point_data, cell_data, cell_type)


def make_mesh(vertices, cells, cell_type):
    '''Mesh from data by MeshEditor'''
    gdim = cell_type.geometric_dimension()
    assert vertices.shape[1] == gdim

    tdim = cell_type.topological_dimension()

    mesh = Mesh()
    editor = MeshEditor()

    editor.open(mesh, str(cell_type), tdim, gdim)            

    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))

    for vi, x in enumerate(vertices): editor.add_vertex(vi, x)

    for ci, c in enumerate(cells): editor.add_cell(ci, c)
    
    editor.close()

    return mesh

"""This file contains dimensions that are needed in ICON.
"""
from dataclasses import dataclass

import numpy as np

from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider


from icon4py.model.common.grid.grid_manager import (
    GridFile,
    GridFileName,
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.grid.simple import SimpleGrid, SimpleGridData


def init_grid_manager(
        fname: str):
    """Creates the a grid based on `netCFD` data.

    Args:
        fname:      The file name to look up.
    """
    from pathlib import Path

    if(not Path(fname).is_file()):
        raise FileNotFoundError(f"The grid to load '{fname}' does not exists.")
    #
    grid_manager = GridManager(ToGt4PyTransformation(), fname, VerticalGridSize(65))
    grid_manager()
    return grid_manager
# end def: init_grid_manager


def compute_e2c2eO(c2e, e2c, with_origin = True)
    dummy_c2e = np.append(c2e, -np.ones_like(c2e), axis=1)
    T = dummy_c2e[e2c, :]
    
    Ts = T.shape
    S = T.reshape(Ts[0], Ts[1] * Ts[2])
    
    if(with_origin):
        # Add the origin index
        SO = np.column_stack((S, np.arange(e2c.shape[0])))
    else:
        SO = S
    #

    
    # Now we get all the unique indexes, we even remove the invalid stuff
    u = []
    for i in range(SO.shape[0]):
        s = S[i]
        s = np.unique(s)
        s = s[s >= 0]
        u.append(s)
    #
    
    highestCount = np.max([len(x)  for x in u])
    
    _e2c2eO = np.zeros((len(u), highestCount))
    
    for i, s in enumerate(u):
        if(len(s) != highestCount):
            diff = highestCount - len(s)
            s = np.append(s, -np.ones(diff))
        #
        _e2c2eO[i] = s
    #

    return _e2c2eO
#



def simple_grid_gridfile(tmp_path):
    path = tmp_path.joinpath(SIMPLE_GRID_NC).absolute()
    grid = SimpleGrid()
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.setncattr(GridFile.PropertyName.GRID_ID, str(uuid4()))
    dataset.createDimension(GridFile.DimensionName.VERTEX_NAME, size=grid.num_vertices)

    dataset.createDimension(GridFile.DimensionName.EDGE_NAME, size=grid.num_edges)
    dataset.createDimension(GridFile.DimensionName.CELL_NAME, size=grid.num_cells)
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE, size=grid.size[E2VDim])
    dataset.createDimension(GridFile.DimensionName.DIAMOND_EDGE_SIZE, size=grid.size[E2C2EDim])
    dataset.createDimension(GridFile.DimensionName.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    dataset.createDimension(GridFile.DimensionName.CELL_GRF, size=14)
    dataset.createDimension(GridFile.DimensionName.EDGE_GRF, size=24)
    dataset.createDimension(GridFile.DimensionName.VERTEX_GRF, size=13)
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_edges),
        GridFile.GridRefinementName.CONTROL_EDGES,
        (GridFile.DimensionName.EDGE_NAME,),
    )

    _add_to_dataset(
        dataset,
        np.zeros(grid.num_cells),
        GridFile.GridRefinementName.CONTROL_CELLS,
        (GridFile.DimensionName.CELL_NAME,),
    )
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_vertices),
        GridFile.GridRefinementName.CONTROL_VERTICES,
        (GridFile.DimensionName.VERTEX_NAME,),
    )

    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE, size=grid.size[C2EDim])
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE, size=grid.size[V2CDim])

    _add_to_dataset(
        dataset,
        grid.connectivities[C2EDim],
        GridFile.OffsetName.C2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[E2CDim],
        GridFile.OffsetName.E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[E2VDim],
        GridFile.OffsetName.E2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[V2CDim],
        GridFile.OffsetName.V2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[C2VDim],
        GridFile.OffsetName.C2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        np.zeros((grid.num_vertices, 4), dtype=np.int32),
        GridFile.OffsetName.V2E2V,
        (GridFile.DimensionName.DIAMOND_EDGE_SIZE, GridFile.DimensionName.VERTEX_NAME),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[V2EDim],
        GridFile.OffsetName.V2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[C2E2CDim],
        GridFile.OffsetName.C2E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    dataset.close()
    yield path
    path.unlink()
# end def: simple_grid_gridfile


from icon4py.model.common.dimension import *







"""This file contains dimensions that are needed in ICON.
"""
from dataclasses import dataclass

import numpy as np

from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider


KDim = Dimension("K", kind=DimensionKind.VERTICAL)
KHalfDim = Dimension("KHalf", kind=DimensionKind.VERTICAL)
EdgeDim = Dimension("Edge")
CellDim = Dimension("Cell")
VertexDim = Dimension("Vertex")
CEDim = Dimension("CE")
CECDim = Dimension("CEC")
ECDim = Dimension("EC")
ECVDim = Dimension("ECV")
CECDim = Dimension("CEC")
CECECDim = Dimension("CECEC")
E2CDim = Dimension("E2C", DimensionKind.LOCAL)
E2VDim = Dimension("E2V", DimensionKind.LOCAL)
C2EDim = Dimension("C2E", DimensionKind.LOCAL)
V2CDim = Dimension("V2C", DimensionKind.LOCAL)
C2VDim = Dimension("C2V", DimensionKind.LOCAL)
V2EDim = Dimension("V2E", DimensionKind.LOCAL)
V2E2VDim = Dimension("V2E2V", DimensionKind.LOCAL)
E2C2VDim = Dimension("E2C2V", DimensionKind.LOCAL)
C2E2CODim = Dimension("C2E2CO", DimensionKind.LOCAL)
E2C2EODim = Dimension("E2C2EO", DimensionKind.LOCAL)
E2C2EDim = Dimension("E2C2E", DimensionKind.LOCAL)
C2E2CDim = Dimension("C2E2C", DimensionKind.LOCAL)
C2E2C2E2CDim = Dimension("C2E2C2E2C", DimensionKind.LOCAL)
E2C = FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
V2C = FieldOffset("V2C", source=CellDim, target=(VertexDim, V2CDim))
C2V = FieldOffset("C2V", source=VertexDim, target=(CellDim, C2VDim))
V2E = FieldOffset("V2E", source=EdgeDim, target=(VertexDim, V2EDim))
E2V = FieldOffset("E2V", source=VertexDim, target=(EdgeDim, E2VDim))
C2CE = FieldOffset("C2CE", source=CEDim, target=(CellDim, C2EDim))
C2CEC = FieldOffset("C2CEC", source=CECDim, target=(CellDim, C2E2CDim))
E2EC = FieldOffset("E2EC", source=ECDim, target=(EdgeDim, E2CDim))
E2ECV = FieldOffset("E2ECV", source=ECVDim, target=(EdgeDim, E2C2VDim))
E2C2V = FieldOffset("E2C2V", source=VertexDim, target=(EdgeDim, E2C2VDim))
C2E2CO = FieldOffset("C2E2CO", source=CellDim, target=(CellDim, C2E2CODim))
E2C2EO = FieldOffset("E2C2EO", source=EdgeDim, target=(EdgeDim, E2C2EODim))
E2C2E = FieldOffset("E2C2E", source=EdgeDim, target=(EdgeDim, E2C2EDim))
C2E2C = FieldOffset("C2E2C", source=CellDim, target=(CellDim, C2E2CDim))
C2E2C2E2C = FieldOffset("C2E2C2E2C", source=CellDim, target=(CellDim, C2E2C2E2CDim))
C2CEC = FieldOffset("C2CEC", source=CECDim, target=(CellDim, C2E2CDim))
C2CECEC = FieldOffset("C2CECEC", source=CECECDim, target=(CellDim, C2E2C2E2CDim))

Koff = FieldOffset("Koff", source=KDim, target=(KDim,))
KHalfOff = FieldOffset("KHalfOff", source=KHalfDim, target=(KHalfDim,))




# periodic
#
# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \2c
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |  3c   \ |   4c  \ |    5c\
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 6c   |  \ 7c   |  \ 8c
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |  9c  \  |  10c \  |  11c \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \12c   |  \ 13c  |  \ 14c
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |  15c  \ | 16c   \ | 17c  \
# 0v       1v         2v        0v


@dataclass
class SimpleGridData:
    c2v_table = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 5],
            [2, 0, 3],
            [0, 3, 4],
            [1, 4, 5],
            [2, 5, 3],
            [3, 4, 7],
            [4, 5, 8],
            [5, 3, 6],
            [3, 6, 7],
            [4, 7, 8],
            [5, 8, 6],
            [6, 7, 1],
            [7, 8, 2],
            [8, 6, 0],
            [6, 0, 1],
            [7, 1, 2],
            [8, 2, 0],
        ]
    )

    e2c2v_table = np.asarray(
        [
            [0, 1, 4, 6],  # 0
            [0, 4, 1, 3],  # 1
            [0, 3, 4, 2],  # 2
            [1, 2, 5, 7],  # 3
            [1, 5, 2, 4],  # 4
            [1, 4, 5, 0],  # 5
            [2, 0, 3, 8],  # 6
            [2, 3, 5, 0],  # 7
            [2, 5, 1, 3],  # 8
            [3, 4, 0, 7],  # 9
            [3, 7, 4, 6],  # 10
            [3, 6, 7, 5],  # 11
            [4, 5, 8, 1],  # 12
            [4, 8, 7, 5],  # 13
            [4, 7, 3, 8],  # 14
            [5, 3, 6, 2],  # 15
            [6, 5, 3, 8],  # 16
            [8, 5, 6, 4],  # 17
            [6, 7, 3, 1],  # 18
            [6, 1, 7, 0],  # 19
            [6, 0, 1, 8],  # 20
            [7, 8, 2, 4],  # 21
            [7, 2, 8, 1],  # 22
            [7, 1, 2, 6],  # 23
            [8, 6, 0, 5],  # 24
            [8, 0, 6, 2],  # 25
            [8, 2, 0, 6],  # 26
        ]
    )

    e2c_table = np.asarray(
        [
            [0, 15],
            [0, 3],
            [3, 2],
            [1, 16],
            [1, 4],
            [0, 4],
            [2, 17],
            [2, 5],
            [1, 5],
            [3, 6],
            [6, 9],
            [9, 8],
            [4, 7],
            [7, 10],
            [6, 10],
            [5, 8],
            [8, 11],
            [7, 11],
            [9, 12],
            [12, 15],
            [15, 14],
            [10, 13],
            [13, 16],
            [12, 16],
            [11, 14],
            [14, 17],
            [13, 17],
        ]
    )

    e2v_table = np.asarray(
        [
            [0, 1],
            [0, 4],
            [0, 3],
            [1, 2],
            [1, 5],
            [1, 4],
            [2, 0],
            [2, 3],
            [2, 5],
            [3, 4],
            [3, 7],
            [3, 6],
            [4, 5],
            [4, 8],
            [4, 7],
            [5, 3],
            [5, 6],
            [5, 8],
            [6, 7],
            [6, 1],
            [6, 0],
            [7, 8],
            [7, 2],
            [7, 1],
            [8, 6],
            [8, 0],
            [8, 2],
        ]
    )

    e2c2e_table = np.asarray(
        [
            [1, 5, 19, 20],
            [0, 5, 2, 9],
            [1, 9, 6, 7],
            [4, 8, 22, 23],
            [3, 8, 5, 12],
            [0, 1, 4, 12],
            [7, 2, 25, 26],
            [6, 2, 8, 15],
            [3, 4, 7, 15],
            [1, 2, 10, 14],
            [9, 14, 11, 18],
            [10, 18, 15, 16],
            [4, 5, 13, 17],
            [12, 17, 14, 21],
            [9, 10, 13, 21],
            [7, 8, 16, 11],
            [15, 11, 17, 24],
            [12, 13, 16, 24],
            [10, 11, 19, 23],
            [18, 23, 20, 0],
            [19, 0, 24, 25],
            [13, 14, 22, 26],
            [21, 26, 23, 3],
            [18, 19, 22, 3],
            [16, 17, 25, 20],
            [24, 20, 26, 6],
            [25, 6, 21, 22],
        ]
    )

    e2c2eO_table = np.asarray(
        [
            [0, 1, 5, 19, 20],
            [0, 1, 5, 2, 9],
            [1, 2, 9, 6, 7],
            [3, 4, 8, 22, 23],
            [3, 4, 8, 5, 12],
            [0, 1, 5, 4, 12],
            [6, 7, 2, 25, 26],
            [6, 7, 2, 8, 15],
            [3, 4, 8, 7, 15],
            [1, 2, 9, 10, 14],
            [9, 10, 14, 11, 18],
            [10, 11, 18, 15, 16],
            [4, 5, 12, 13, 17],
            [12, 13, 17, 14, 21],
            [9, 10, 14, 13, 21],
            [7, 8, 15, 16, 11],
            [15, 16, 11, 17, 24],
            [12, 13, 17, 16, 24],
            [10, 11, 18, 19, 23],
            [18, 19, 23, 20, 0],
            [19, 20, 0, 24, 25],
            [13, 14, 21, 22, 26],
            [21, 22, 26, 23, 3],
            [18, 19, 23, 22, 3],
            [16, 17, 24, 25, 20],
            [24, 25, 20, 26, 6],
            [25, 26, 6, 21, 22],
        ]
    )

    c2e_table = np.asarray(
        [
            [0, 1, 5],  # cell 0
            [3, 4, 8],  # cell 1
            [6, 7, 2],  # cell 2
            [1, 2, 9],  # cell 3
            [4, 5, 12],  # cell 4
            [7, 8, 15],  # cell 5
            [9, 10, 14],  # cell 6
            [12, 13, 17],  # cell 7
            [15, 16, 11],  # cell 8
            [10, 11, 18],  # cell 9
            [13, 14, 21],  # cell 10
            [16, 17, 24],  # cell 11
            [18, 19, 23],  # cell 12
            [21, 22, 26],  # cell 13
            [24, 25, 20],  # cell 14
            [19, 20, 0],  # cell 15
            [22, 23, 3],  # cell 16
            [25, 26, 6],  # cell 17
        ]
    )

    v2c_table = np.asarray(
        [
            [17, 14, 3, 0, 2, 15],
            [0, 4, 1, 12, 16, 15],
            [1, 5, 2, 16, 13, 17],
            [3, 6, 9, 5, 8, 2],
            [6, 10, 7, 4, 0, 3],
            [7, 11, 8, 5, 1, 4],
            [9, 12, 15, 8, 11, 14],
            [12, 16, 13, 10, 6, 9],
            [13, 17, 14, 11, 7, 10],
        ]
    )

    v2e_table = np.asarray(
        [
            [0, 1, 2, 6, 25, 20],
            [3, 4, 5, 0, 23, 19],
            [6, 7, 8, 3, 22, 26],
            [9, 10, 11, 15, 7, 2],
            [12, 13, 14, 9, 1, 5],
            [15, 16, 17, 12, 4, 8],
            [18, 19, 20, 24, 16, 11],
            [21, 22, 23, 18, 10, 14],
            [24, 25, 26, 21, 13, 17],
        ]
    )

    diamond_table = np.asarray(
        [
            [0, 1, 4, 6],  # 0
            [0, 4, 1, 3],
            [0, 3, 4, 2],
            [1, 2, 5, 7],  # 3
            [1, 5, 2, 4],
            [1, 4, 5, 0],
            [2, 0, 3, 8],  # 6
            [2, 3, 0, 5],
            [2, 5, 1, 3],
            [3, 4, 0, 7],  # 9
            [3, 7, 4, 6],
            [3, 6, 5, 7],
            [4, 5, 1, 8],  # 12
            [4, 8, 5, 7],
            [4, 7, 3, 8],
            [5, 3, 2, 6],  # 15
            [5, 6, 3, 8],
            [5, 8, 4, 6],
            [6, 7, 3, 1],  # 18
            [6, 1, 7, 0],
            [6, 0, 1, 8],
            [7, 8, 4, 2],  # 21
            [7, 2, 8, 1],
            [7, 1, 6, 2],
            [8, 6, 5, 0],  # 24
            [8, 0, 6, 2],
            [8, 2, 7, 0],
        ]
    )

    c2e2cO_table = np.asarray(
        [
            [15, 4, 3, 0],
            [16, 5, 4, 1],
            [17, 3, 5, 2],
            [0, 6, 2, 3],
            [1, 7, 0, 4],
            [2, 8, 1, 5],
            [3, 10, 9, 6],
            [4, 11, 10, 7],
            [5, 9, 11, 8],
            [6, 12, 8, 9],
            [7, 13, 6, 10],
            [8, 14, 7, 11],
            [9, 16, 15, 12],
            [10, 17, 16, 13],
            [11, 15, 17, 14],
            [12, 0, 14, 15],
            [13, 1, 12, 16],
            [14, 2, 13, 17],
        ]
    )

    c2e2c_table = np.asarray(
        [
            [15, 4, 3],
            [16, 5, 4],
            [17, 3, 5],
            [0, 6, 2],
            [1, 7, 0],
            [2, 8, 1],
            [3, 10, 9],
            [4, 11, 10],
            [5, 9, 11],
            [6, 12, 8],
            [7, 13, 6],
            [8, 14, 7],
            [9, 16, 15],
            [10, 17, 16],
            [11, 15, 17],
            [12, 0, 14],
            [13, 1, 12],
            [14, 2, 13],
        ]
    )

    c2e2c2e2c_table = np.asarray(
        [
            [15, 4, 3, 12, 14, 1, 7, 6, 2],  # 1c
            [16, 5, 4, 12, 13, 2, 8, 7, 0],
            [17, 3, 5, 13, 14, 0, 6, 8, 1],
            [0, 6, 2, 17, 5, 9, 10, 15, 4],
            [1, 7, 0, 15, 3, 16, 5, 10, 11],  # 5c
            [2, 8, 1, 4, 16, 17, 3, 9, 11],
            [3, 10, 9, 2, 0, 7, 13, 8, 12],
            [4, 11, 10, 0, 1, 8, 14, 6, 13],
            [5, 9, 11, 1, 2, 3, 12, 7, 14],
            [6, 12, 8, 5, 11, 3, 10, 16, 15],  # 10c
            [7, 13, 6, 3, 9, 4, 11, 16, 17],
            [8, 14, 7, 4, 10, 5, 9, 15, 17],
            [9, 16, 15, 8, 6, 1, 13, 0, 14],
            [10, 17, 16, 6, 7, 2, 14, 1, 12],
            [11, 15, 17, 7, 8, 2, 13, 0, 12],  # 15c
            [12, 0, 14, 11, 17, 9, 16, 3, 4],
            [13, 1, 12, 9, 15, 10, 17, 4, 5],
            [14, 2, 13, 10, 16, 5, 3, 11, 15],
        ]
    )
#

@dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int
#

@dataclass(frozen=True)
class VerticalGridSize:
    num_lev: int
#

@dataclass(frozen=True)
class GridConfig:
    horizontal_config: HorizontalGridSize
    vertical_config: VerticalGridSize
    limited_area: bool = True
    n_shift_total: int = 0
    lvertnest: bool = False

    @property
    def num_levels(self):
        return self.vertical_config.num_lev

    @property
    def num_vertices(self):
        return self.horizontal_config.num_vertices

    @property
    def num_edges(self):
        return self.horizontal_config.num_edges

    @property
    def num_cells(self):
        return self.horizontal_config.num_cells
#


class SimpleGrid:
    _CELLS = 18
    _EDGES = 27
    _VERTICES = 9

    _KLEVELS = 10       # My extension.

    def __init__(self):
        """Instantiate a SimpleGrid used for testing purposes."""
        if(True):
            self.config: GridConfig = None
            self.connectivities: Dict[Dimension, np.ndarray] = {}
            self.size: Dict[Dimension, int] = {}
            self.offset_provider_mapping: Dict[str, tuple[Callable, Dimension, ...]] = {}
        #

        self._configure()
        self.offset_provider_mapping = {
            "C2E": (self._get_offset_provider, C2EDim, CellDim, EdgeDim),
            "C2E2CO": (self._get_offset_provider, C2E2CODim, CellDim, CellDim),
            "C2E2C": (self._get_offset_provider, C2E2CDim, CellDim, CellDim),
            "E2C2EO": (self._get_offset_provider, E2C2EODim, EdgeDim, EdgeDim),
            "E2C2E": (self._get_offset_provider, E2C2EDim, EdgeDim, EdgeDim),
            "V2C": (self._get_offset_provider, V2CDim, VertexDim, CellDim),
            "V2E": (self._get_offset_provider, V2EDim, VertexDim, EdgeDim),
            "E2C": (self._get_offset_provider, E2CDim, EdgeDim, CellDim),
            "E2V": (self._get_offset_provider, E2VDim, EdgeDim, VertexDim),
            "E2C2V": (self._get_offset_provider, E2C2VDim, EdgeDim, VertexDim),
            #"C2CE": (self._get_offset_provider_for_sparse_fields, C2EDim, CellDim, CEDim),
            "Koff": (lambda: KDim,),  # Koff is a special case
            "C2E2C2E2C": (self._get_offset_provider, C2E2C2E2CDim, CellDim, CellDim),
            #"E2ECV": (self._get_offset_provider_for_sparse_fields, E2C2VDim, EdgeDim, ECVDim),
            #"E2EC": (self._get_offset_provider_for_sparse_fields, E2CDim, EdgeDim, ECDim),
            #"C2CEC": (self._get_offset_provider_for_sparse_fields, C2E2CDim, CellDim, CECDim),
        }
    #


    @property
    def num_cells(self) -> int:
        return self.config.num_cells
        #return self._CELLS
    #


    @property
    def num_vertices(self) -> int:
        return self.config.num_vertices
        #return self._VERTICES
    #


    @property
    def num_edges(self) -> int:
        return self_config.num_edges
        #return self._EDGES
    #


    @property
    def diamond_table(self) -> int:
        return SimpleGridData.diamond_table
        #raise NotImplementedError
    #


    @property
    def num_levels(self) -> int:
        return self.config.num_levels
        #return self._KLEVELS
    #


    def _configure(self):
        horizontal_grid_size = HorizontalGridSize(
            num_vertices=self._VERTICES, num_edges=self._EDGES, num_cells=self._CELLS
        )
        vertical_grid_size = VerticalGridSize(num_lev=self._KLEVELS)
        config = GridConfig(
            horizontal_config=horizontal_grid_size,
            vertical_config=vertical_grid_size,
        )

        connectivity_dict = {
            C2VDim: SimpleGridData.c2v_table,
            E2CDim: SimpleGridData.e2c_table,
            E2VDim: SimpleGridData.e2v_table,
            C2EDim: SimpleGridData.c2e_table,
            C2E2CODim: SimpleGridData.c2e2cO_table,
            C2E2CDim: SimpleGridData.c2e2c_table,
            E2C2EODim: SimpleGridData.e2c2eO_table,
            E2C2EDim: SimpleGridData.e2c2e_table,
            E2C2VDim: SimpleGridData.e2c2v_table,
            V2CDim: SimpleGridData.v2c_table,
            V2EDim: SimpleGridData.v2e_table,
            C2E2C2E2CDim: SimpleGridData.c2e2c2e2c_table,
        }

        self.with_config(config).with_connectivities(connectivity_dict)
        self.update_size_connectivities(
            {
                ECVDim: self.size[EdgeDim] * self.size[E2C2VDim],
                CEDim: self.size[CellDim] * self.size[C2EDim],
                ECDim: self.size[EdgeDim] * self.size[E2CDim],
            }
        )
    #


    def update_size_connectivities(self, new_sizes):
        self.size.update(new_sizes)
    #


    def get_offset_provider(self, name):
        if name in self.offset_provider_mapping:
            method, *args = self.offset_provider_mapping[name]
            return method(*args)
        else:
            raise Exception(f"Offset provider for {name} not found.")
    #


    def _get_offset_provider(self, dim, from_dim, to_dim):
        return NeighborTableOffsetProvider(
            self.connectivities[dim], from_dim, to_dim, self.size[dim]
        )
    #


    def _get_offset_provider_for_sparse_fields(self, dim, from_dim, to_dim):
        return neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities[dim].shape, from_dim, to_dim
        )
    #
#

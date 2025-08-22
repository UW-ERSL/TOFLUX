"""Simple geometry Mesher."""

import dataclasses
import chex
from typing import Optional, Tuple, Union
import numpy as np
import scipy.spatial as spy_spatial
import jax
import jax.numpy as jnp
import shapely.geometry as shap_geom

import toflux.src.utils as _utils
import toflux.src.geometry as _geom
import toflux.src.element as _element


@chex.dataclass
class BoundingBox:
  """Defines a bounding box in 1D, 2D or 3D."""
  x: _utils.Extent
  y: Optional[_utils.Extent] = None
  z: Optional[_utils.Extent] = None

  @property
  def num_dim(self) -> int:
    """The number of dimensions of the bounding box."""
    return 1 + int(self.y is not None) + int(self.z is not None)

  @property
  def min(self) -> jnp.ndarray:
    """The minimum coordinates of the bounding box."""
    return jnp.array(
      [self.x.min, self.y.min if self.y else 0.0, self.z.min if self.z else 0.0]
    )[: self.num_dim]

  @property
  def max(self) -> jnp.ndarray:
    """The maximum coordinates of the bounding box."""
    return jnp.array(
      [self.x.max, self.y.max if self.y else 0.0, self.z.max if self.z else 0.0]
    )[: self.num_dim]

  @property
  def lx(self) -> float:
    """The length of the bounding box in the x direction."""
    return jnp.abs(self.x.range)

  @property
  def ly(self) -> float:
    """The length of the bounding box in the y direction."""
    return jnp.abs(self.y.range) if self.y else 0.0

  @property
  def lz(self) -> float:
    """The length of the bounding box in the z direction."""
    return jnp.abs(self.z.range) if self.z else 0.0

  @property
  def diag_length(self) -> float:
    """The length of the diagonal of the bounding box."""
    return jnp.sqrt(self.lx**2 + self.ly**2 + self.lz**2)


@dataclasses.dataclass
class Nodes:
  """Represents nodes in a finite element mesh.

  Attributes:
    coords: Array of (num_nodes, num_dim) of the spatial coordinate of the nodes.
    dof_per_node: The number of degrees of freedom at the node.
  """

  coords: jax.Array
  dof_per_node: Optional[int] = 0

  @property
  def num_nodes(self) -> int:
    return self.coords.shape[0]

  @property
  def num_dim(self) -> int:
    return self.coords.shape[1]

  def __post_init__(self):
    """Post-initialization to set additional properties."""
    assert self.num_dim > 0, "Number of dimensions must be greater than 0"
    assert self.coords.ndim == 2, "Coordinates must be of shape (num_nodes, num_dim)"


@dataclasses.dataclass
class Mesh:
  """Base class for finite element mesh.

  Attributes:
    nodes: `Nodes` that compose the mesh.
    elem_nodes: Array of integers of (num_elems, nodes_per_elem) that contain the
       global node numbers of the elements.
    elem_template: The isoparametric element used in the mesh.
    gauss_order: (Optional) The order of the Gauss quadrature used in the mesh.

  Derived Attributes:
    num_dim: (int) The number of dimensions of the mesh.
    num_nodes: (int) The number of nodes in the mesh.
    num_elems: (int) The number of elements in the mesh.
    num_dofs: (int) The total number of degrees of freedom in the mesh.
    dofs_per_elem: (int) The number of degrees of freedom per element.
    bounding_box: The bounding box of the mesh.
    elem_centers: Array of (num_elems, num_dim) of the  center of the elements.
    mesh_kdtree: The KDTree of the mesh.
  """

  nodes: Nodes
  elem_nodes: np.ndarray
  elem_template: Union[_element.Cube8, _element.Rect4]
  gauss_order: Optional[int] = None

  def _get_bounding_box(self) -> BoundingBox:
    """Get the bounding box of the mesh."""
    min_xy = jnp.min(self.nodes.coords, axis=0)
    max_xy = jnp.max(self.nodes.coords, axis=0)
    return BoundingBox(
      x=_utils.Extent(min=min_xy[0], max=max_xy[0]),
      y=_utils.Extent(min=min_xy[1], max=max_xy[1]) if len(min_xy) > 1 else None,
      z=_utils.Extent(min=min_xy[2], max=max_xy[2]) if len(min_xy) > 2 else None,
    )

  def _compute_elem_coords(self) -> Tuple[jax.Array, jax.Array]:
    """Compute the element centers."""
    elem_centers = jnp.mean(self.nodes.coords[self.elem_nodes], axis=1)
    elem_node_coords = jnp.array([self.nodes.coords[elem] for elem in self.elem_nodes])
    return elem_centers, elem_node_coords

  def _compute_elem_volume(self) -> Tuple[jax.Array, float]:
    """Compute the volume of each element."""
    elem_volume = jax.vmap(self.elem_template.elem_volume)(self.elem_node_coords)
    domain_volume = jnp.sum(elem_volume)
    return elem_volume, domain_volume

  def _compute_connectivity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the element connectivity matrices."""
    elem_dof_mat = np.array(
      [
        self.nodes.dof_per_node * node + j
        for elem_nodes in self.elem_nodes
        for node in elem_nodes
        for j in range(self.nodes.dof_per_node)
      ]
    ).reshape(self.num_elems, self.num_dofs_per_elem)

    matrx_size = self.num_elems * self.num_dofs_per_elem**2
    iK = np.kron(
      elem_dof_mat, np.ones((self.num_dofs_per_elem, 1), dtype=int)
    ).T.reshape(matrx_size, order="F")
    jK = np.kron(
      elem_dof_mat, np.ones((1, self.num_dofs_per_elem), dtype=int)
    ).T.reshape(matrx_size, order="F")
    return elem_dof_mat, iK, jK

  def _identify_all_elem_face_nodes(self) -> np.ndarray:
    """
    Return an array of shape
        (num_elems, num_faces_per_elem, num_nodes_per_face)
    containing the global node numbers for every face of every element.
    """
    face_idx = np.asarray(self.elem_template.face_connectivity, dtype=np.int64)
    return self.elem_nodes[:, face_idx]

  @staticmethod
  def identify_all_boundary_elem_faces(elem_face_nodes: np.ndarray) -> np.ndarray:
    """Identifies all boundary faces of the mesh. A boundary face is one that occurs only
      once in the mesh (under any permutation). Thus, we count all the occurences of the
      nodes that make up a face and return a boolean array with True if the occurance
      is unique.

    Args:
      elem_face_nodes: An array of (num_elems, num_faces_per_elem, num_nodes_per_face) of
        the nodes of each face.

    Returns: A boolean array of (num_elems, num_faces_per_elem) that indicates if the face
      is on the boundary.
    """
    num_elems, num_faces_per_elem, num_nodes_per_face = elem_face_nodes.shape
    sorted_faces = np.sort(elem_face_nodes, axis=2)
    all_faces = sorted_faces.reshape(-1, num_nodes_per_face)
    _, unique_inverse, counts = np.unique(
      all_faces, axis=0, return_inverse=True, return_counts=True
    )
    boundary_mask = counts == 1
    return boundary_mask[unique_inverse].reshape(num_elems, num_faces_per_elem)

  def __post_init__(self):
    """Post-initialization to set additional properties."""

    self.num_dim = self.nodes.num_dim
    self.num_nodes = self.nodes.num_nodes
    self.num_elems = self.elem_nodes.shape[0]
    self.num_dofs = self.nodes.dof_per_node * self.num_nodes
    self.num_dofs_per_elem = self.nodes.dof_per_node * self.elem_template.num_nodes

    self.gauss_pts, self.gauss_weights = _utils.gauss_integ_points_weights(
      order=self.gauss_order, dimension=self.num_dim
    )

    self.bounding_box = self._get_bounding_box()
    self.elem_centers, self.elem_node_coords = self._compute_elem_coords()
    self.mesh_kdtree = spy_spatial.KDTree(self.elem_centers)
    self.elem_dof_mat, self.iK, self.jK = self._compute_connectivity()
    self.elem_volume, self.domain_volume = self._compute_elem_volume()
    self.elem_diag_length = jax.vmap(self.elem_template.diag_length)(
      self.elem_node_coords
    )

    self.elem_face_nodes = self._identify_all_elem_face_nodes()
    self.boundary_faces = self.identify_all_boundary_elem_faces(self.elem_face_nodes)

    assert (
      jnp.max(self.elem_nodes) == self.num_nodes - 1
    ), "Element nodes refer to a non-existent node"

  def get_nearest_elem(self, points: jax.Array) -> np.ndarray:
    """Get the nearest element index of query points.

    The nearest point is derived from querying a kdtree constructed from the element
    centers.

    NOTE: No checks are performed to see if the query points are within the bounding box
    of the mesh. In case the point is outside the bounding box, the nearest element
    index is still returned.

    Args:
      points: Array of (num_points, num_dim) of the query points.
    Returns: Array of (num_points,) of the nearest element index.
    """
    return self.mesh_kdtree.query(points)[1]


class VoxelMesh(Mesh):
  """Mesh defined by voxels (3D) or pixels (2D).

  This is a special instance of a mesh where the nodes are aligned in a grid pattern.
  The elements are defined by the voxel or pixel connectivity.

  Attributes:
    nodes: `Nodes` that compose the mesh.
    elem_nodes: Array of integers of (num_elems, nodes_per_elem) that contain the
       global node numbers of the elements.
    elem_template: The isoparametric element used in the mesh.
    gauss_order: The order of the Gauss quadrature used in the mesh.
    elem_size: An array of (num_dim,) containing the size of the elements.
  """

  def __init__(
    self,
    nodes: Nodes,
    elem_nodes: np.ndarray,
    elem_template: Union[_element.Cube8, _element.Rect4],
    elem_size: jax.Array,
    gauss_order: Optional[int] = None,
  ):
    super().__init__(nodes, elem_nodes, elem_template, gauss_order)
    self.elem_size = elem_size
    self._check_if_voxel_mesh()

  def _check_if_voxel_mesh(self, tol: float = 0.05):
    """Check if the mesh is a voxel mesh.

    The function ensures that the mesh initialized using this class is indeed a voxel
    mesh.

    The following checks are performed:
      1. The element template is either Hex8 or Quad4.
      2. All the element sizes are the same.
      TODO: 3. All the elements are aligned in a grid pattern.

    Args:
      tol: The tolerance used to check if the elements are the same size.

    Raises: ValueError if the mesh is not a voxel mesh.
    """
    # Check if the element template is either Line2/ Rect4 or Hex8
    if not (
      isinstance(self.elem_template, _element.Cube8)
      or isinstance(self.elem_template, _element.Rect4)
    ):
      raise ValueError("Voxel mesh must have Line2 or Cube8 or Rect4 elements.")

    # Check if all the elements are the same size
    min_coord = jnp.amin(self.elem_node_coords, axis=1)  # (num_elems, num_dim)
    max_coord = jnp.amax(self.elem_node_coords, axis=1)
    elem_sizes = max_coord - min_coord
    reference_size = elem_sizes[0, :]
    if not np.all(
      np.linalg.norm(elem_sizes - reference_size, axis=1) < tol * reference_size[0]
    ):
      raise ValueError("All elements must be the same size.")


def deform_mesh(mesh: Mesh, delta: jax.Array) -> Mesh:
  """Displace the nodes of the mesh by the `delta` displacement.

  NOTE: No checks are performed to ensure the delta is of the correct shape.

  NOTE: No checks are performed to ensure that the nodes do not result in elements
    that may self intersect or become invalid in any sense.

  Args:
    delta: Array of (num_nodes, num_dim) of the displacement of the nodes.

  Returns: The deformed mesh.
  """
  coords = mesh.nodes.coords + delta
  return Mesh(
    nodes=Nodes(coords=coords, dof_per_node=mesh.nodes.dof_per_node),
    elem_nodes=mesh.elem_nodes,
    elem_template=mesh.elem_template,
    gauss_order=mesh.gauss_order,
  )


def get_dofs_of_nodes(mesh: Mesh, nodes: np.ndarray) -> np.ndarray:
  """Get the degrees of freedom of the nodes.

  Args:
    mesh: The mesh object.
    nodes: Array of (num_nodes,) containing the node indices.

  Returns: Array of (num_nodes, num_dofs_per_node) containing the degrees of freedom
    of the nodes.
  """
  return np.array(
    [
      mesh.nodes.dof_per_node * node + j
      for node in nodes
      for j in range(mesh.nodes.dof_per_node)
    ]
  ).reshape(nodes.shape[0], mesh.nodes.dof_per_node)


def compute_point_indices_in_box(coords: np.ndarray, bbox: BoundingBox):
  """Filters the coordinates in `xy` that are within the bounding box.

  Args:
    coords: An array of shape (num_pts, num_dim) containing the coordinates.
    bbox: Defines the coordinates of the bounding box.

  Returns: A Boolean array of shape (num_pts,) with True values for indices
    whose coordinates are within the bounding box.
  """
  filtered_indices = jnp.ones(coords.shape[0], dtype=bool)

  x_in_box = jnp.logical_and(coords[:, 0] >= bbox.x.min, coords[:, 0] <= bbox.x.max)
  filtered_indices = jnp.logical_and(filtered_indices, x_in_box)

  if bbox.y is not None:
    y_in_box = jnp.logical_and(coords[:, 1] >= bbox.y.min, coords[:, 1] <= bbox.y.max)
    filtered_indices = jnp.logical_and(filtered_indices, y_in_box)
  if bbox.z is not None:
    z_in_box = jnp.logical_and(coords[:, 2] >= bbox.z.min, coords[:, 2] <= bbox.z.max)
    filtered_indices = jnp.logical_and(filtered_indices, z_in_box)
  return filtered_indices


class GridMesh(VoxelMesh):
  """Structured grid mesher in 1D, 2D and 3D.

  The mesh is created by dividing the domain as defined by the bounding box into
  rectangular elements. This is a specific implementation of the VoxelMesh class
  where the number of elements along each dimension is specified and is regular. In
  other words, in addition to the attributes of the VoxelMesh class, this class
  also has the number of elements along each dimension.
  """

  def __init__(
    self,
    nel: Tuple[int, Optional[int], Optional[int]],
    bounding_box: BoundingBox,
    dofs_per_node: Optional[int] = 0,
    gauss_order: Optional[int] = None,
  ):
    num_dim = len(nel)
    nel = np.array(nel).astype(np.int32)
    num_elems = np.prod(nel)
    grid_shape = nel + 1

    min_coords = bounding_box.min
    max_coords = bounding_box.max
    dom_size = max_coords - min_coords
    elem_size = dom_size / nel

    self.nelx = nel[0]
    if num_dim > 1:
      self.nely = nel[1]
    if num_dim > 2:
      self.nelz = nel[2]

    coords = [
      np.linspace(min_coords[d], max_coords[d], grid_shape[d]) for d in range(num_dim)
    ]
    grids = np.meshgrid(*coords, indexing="ij")
    node_coords = np.stack(grids, axis=-1).reshape(-1, num_dim)
    nodes = Nodes(coords=node_coords, dof_per_node=dofs_per_node)

    elem_indices_ranges = [np.arange(n) for n in nel]
    elem_indices_grid = np.meshgrid(*elem_indices_ranges, indexing="ij")
    elem_indices_flat = [g.flatten() for g in elem_indices_grid]
    base_node_indices = np.zeros(num_elems, dtype=jnp.int32)
    stride = 1
    for d in range(num_dim - 1, -1, -1):
      base_node_indices += elem_indices_flat[d] * stride
      stride *= grid_shape[d]

    if num_dim == 2:
      elem_template = _element.Rect4()
      nodes_y = grid_shape[1]
      offsets = np.array([0, nodes_y, nodes_y + 1, 1])
    elif num_dim == 3:
      elem_template = _element.Cube8()
      nodes_y, nodes_z = grid_shape[1], grid_shape[2]
      nodes_yz = nodes_y * nodes_z
      offsets = np.array(
        [
          0,  # (i,   j,   k   )  local‐0
          nodes_yz,  # (i+1, j,   k   )  local‐1
          nodes_yz + nodes_z,  # (i+1, j+1, k   )  local‐2
          nodes_z,  # (i,   j+1, k   )  local‐3
          1,  # (i,   j,   k+1 )  local‐4
          nodes_yz + 1,  # (i+1, j,   k+1 )  local‐5
          nodes_yz + nodes_z + 1,  # (i+1, j+1, k+1 )  local‐6
          nodes_z + 1,  # (i,   j+1, k+1 )  local‐7
        ]
      )
    else:
      raise ValueError("Only 2D, and 3D are supported")
    elem_nodes = base_node_indices[:, None] + offsets
    super().__init__(
      nodes=nodes,
      elem_nodes=elem_nodes,
      elem_template=elem_template,
      elem_size=elem_size,
      gauss_order=gauss_order,
    )


def grid_mesh_brep(
  brep: _geom.BrepGeometry,
  nelx_desired: int,
  nely_desired: int,
  dofs_per_node: int,
  gauss_order: int,
) -> VoxelMesh:
  """Create a structured grid mesh from a boundary representation file.

  NOTE: The number of elements `nelx` and `nely` are not strictly enforced. The
  function will create a mesh with the number of elements closest to the desired
  number of elements. The values provided are used as a guide to define the outermost
  bounding box from which elements are removed.

  Args:
    brep_file: The path to the boundary representation file (geojson).
    nelx_desired: The number of elements in the x-direction desired.
    nely_desired: The number of elements in the y-direction desired.
    dofs_per_node: The number of degrees of freedom per node.
    gauss_order: The order of the Gauss quadrature used in the mesh.
  """

  min_x, min_y, max_x, max_y = brep.geometry.bounds
  x_coords = np.linspace(min_x, max_x, nelx_desired + 1)
  y_coords = np.linspace(min_y, max_y, nely_desired + 1)
  xx, yy = np.meshgrid(x_coords, y_coords)

  nodes = np.vstack([xx.ravel(), yy.ravel()]).T
  elem_size = np.abs(np.array([x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]]))
  elem_nodes = []
  for i in range(nely_desired):
    for j in range(nelx_desired):
      n1 = i * (nelx_desired + 1) + j
      n2 = n1 + 1
      n3 = n1 + (nelx_desired + 1)
      n4 = n3 + 1
      elem_nodes.append([n1, n2, n4, n3])
  elem_nodes = np.array(elem_nodes)

  valid_elems = []
  for elem in elem_nodes:
    elem_centroid = nodes[elem].mean(axis=0)
    if brep.geometry.contains(shap_geom.Point(elem_centroid)):
      valid_elems.append(elem)
  valid_elems = np.array(valid_elems)

  used_nodes = np.unique(valid_elems)
  node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_nodes)}

  filtered_nodes = nodes[used_nodes]
  filtered_elems = np.array([[node_map[idx] for idx in elem] for elem in valid_elems])

  nodes = Nodes(coords=filtered_nodes, dof_per_node=dofs_per_node)
  return VoxelMesh(
    nodes=nodes,
    elem_nodes=filtered_elems,
    elem_template=_element.Rect4(),
    elem_size=elem_size,
    gauss_order=gauss_order,
  )

"""Quad and Hexahedral element templates."""

import jax
import jax.numpy as jnp


class Quad4:
  """Quad element with 4 nodes.

  The nodes are numbered as follows:

        y                          3---c----2
          |                         \       |
          |                        d \      |b
          ---->x                      0---a--1
  """

  @property
  def dimension(self) -> int:
    return 2

  @property
  def num_nodes(self) -> int:
    return 4

  @property
  def meshio_type(self) -> str:
    return "quad"

  @property
  def num_edges(self) -> int:
    return 4

  @property
  def num_faces(self) -> int:
    return 4

  @property
  def num_nodes_per_face(self) -> int:
    return 2

  @property
  def edge_connectivity(self) -> jax.Array:
    """Connectivity of the edges of the element.

    Returns: An array of shape (num_edges, nodes_per_edge) containing the connectivity of
      the edges of the element. Here nodes_per_edge = 2. Hence, the first column
      contains the start node of the edge and the second column contains the end node of
      the edge. The edges are numbered so that the face normal points outwards.
    """
    return jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])

  @property
  def face_connectivity(self) -> jax.Array:
    """Connectivity of the faces of the element.

    NOTE: For a quad element, the face is the same as the edge.

    Returns: An array of shape (num_faces, nodes_per_face) containing the connectivity of
      the faces of the element. Here nodes_per_face = 2. Hence, the first column
      contains the start node of the face and the second column contains the end node of
      the face. The edges are numbered so that the face normal points outwards.
    """
    return jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])

  @staticmethod
  def diag_length(node_coords: jax.Array) -> float:
    """Diagonal length of the element.

    Compute the average of the main and secondary diagonals of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The average diagonal length of the element.
    """
    diag_main = jnp.linalg.norm(node_coords[0, :] - node_coords[2, :])
    diag_secondary = jnp.linalg.norm(node_coords[1, :] - node_coords[3, :])
    return 0.5 * (diag_main + diag_secondary)

  @staticmethod
  def elem_volume(node_coords: jax.Array) -> jax.Array:
    """Volume (area for quad in 2D; length in 1D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume (area) of the element.
    """
    # For a quad element, we can compute the area by dividing it into two triangles.
    # Using shoelace formula for each triangle and sum the areas.

    # Triangle 1: Nodes 0, 1, 2
    area1 = 0.5 * (
      node_coords[0, 0] * (node_coords[1, 1] - node_coords[2, 1])
      + node_coords[1, 0] * (node_coords[2, 1] - node_coords[0, 1])
      + node_coords[2, 0] * (node_coords[0, 1] - node_coords[1, 1])
    )

    # Triangle 2: Nodes 0, 2, 3
    area2 = 0.5 * (
      node_coords[0, 0] * (node_coords[2, 1] - node_coords[3, 1])
      + node_coords[2, 0] * (node_coords[3, 1] - node_coords[0, 1])
      + node_coords[3, 0] * (node_coords[0, 1] - node_coords[2, 1])
    )
    return jnp.abs(area1) + jnp.abs(area2)

  @staticmethod
  def face_normal(face_node_coords: jax.Array) -> jax.Array:
    """Normal of the faces of the element.

    Args:
      face_node_coords: Array of shape (nodes_per_face, num_dim) containing the
       coordinates of the nodes of the face. In this instance, the nodes_per_face = 2
       and num_dim is 2.

    Returns: An array of shape (num_dim,) containing the normal of a face.
    """
    edge = face_node_coords[1, :] - face_node_coords[0, :]
    normal = jnp.array([-edge[1], edge[0]])
    normal /= jnp.linalg.norm(normal)
    return normal

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the isoparametric coordinate of a point in the element.
    NOTE: No checks are performed to ensure that the point is within the element.
    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1].
    """
    raise NotImplementedError("Not implemented for Quad4 element.")

  @staticmethod
  def shape_functions(
    isoparam_pt: jax.Array,
  ) -> jax.Array:
    """Compute the shape functions at the given xi, eta.

    Args:
      isoparam_pt: Array of (2,) of the isoparametric xi, and eta coordinate.

    Returns: Array of shape (4,) containing the shape functions at the given
      xi, and eta. The 4 corresponds to the number of nodes in the element.
    """
    return jnp.array(
      [
        0.25 * (1 - isoparam_pt[0]) * (1 - isoparam_pt[1]),
        0.25 * (1 + isoparam_pt[0]) * (1 - isoparam_pt[1]),
        0.25 * (1 + isoparam_pt[0]) * (1 + isoparam_pt[1]),
        0.25 * (1 - isoparam_pt[0]) * (1 + isoparam_pt[1]),
      ]
    )

  @staticmethod
  def get_physical_coordinate(
    isoparam_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the physical coordinate of a point in the element.

    Args:
      isoparam_coordn: Array of shape (num_dim,) containing the isoparametric
        coordinates of the point. The values are in the range [-1, 1].
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the physical coordinate of the
      point.
    """
    shp_fn = Quad4.shape_functions(isoparam_coordn)
    return jnp.einsum("n, nd -> d", shp_fn, node_coords)

  @staticmethod
  def shape_function_gradients_isoparametric(
    gauss_pt: jax.Array,
  ) -> jax.Array:
    """Compute the shape function gradients at the given xi, and eta.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinates.
        Here, 2 corresponds to the number of isoparametric dimensions.

    Returns: Array of (4, 2) arrays containing the shape function gradients at
      given xi, and eta. The 4 corresponds to the number of nodes in the element
      and 2 corresponds to the number of physical dimensions.
    """
    xis = jnp.array([-1, 1, 1, -1])
    etas = jnp.array([-1, -1, 1, 1])
    dN_dxi = jnp.array(
      [
        0.25 * xis[0] * (1 - gauss_pt[1]),
        0.25 * xis[1] * (1 - gauss_pt[1]),
        0.25 * xis[2] * (1 + gauss_pt[1]),
        0.25 * xis[3] * (1 + gauss_pt[1]),
      ]
    )
    dN_deta = jnp.array(
      [
        0.25 * (1 - gauss_pt[0]) * etas[0],
        0.25 * (1 + gauss_pt[0]) * etas[1],
        0.25 * (1 + gauss_pt[0]) * etas[2],
        0.25 * (1 - gauss_pt[0]) * etas[3],
      ]
    )
    return jnp.stack((dN_dxi, dN_deta), axis=1)

  @classmethod
  def compute_jacobian_and_determinant(
    cls,
    gauss_pt: jax.Array,
    node_coords: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Compute the Jacobian of the element at the given xi, and eta.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinate.
      node_coords: Array of (4, 2) containing the x, and y coordinates of the nodes.
        The four corresponds to the number of nodes and 2 corresponds to the number
        of physical dimensions.

    Returns: A tuple containing the Jacobian and the its determinant:
      - Jacobians of shape (2, 2) containing the Jacobian of the element at
          the given xi, and eta. 2 corresponds to the number of physical dims.
      - Determinant of shape (1,) containing the determinant of the Jacobian.
    """
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)
    # num_(n)odes, (d)[i]m
    jac = jnp.einsum("nd, ni -> di", gradN_isoparam, node_coords)
    det_jac = jnp.linalg.det(jac)
    return jac, det_jac

  @classmethod
  def get_gradient_shape_function_physical(
    cls,
    gauss_pt: jax.Array,
    node_coords: jax.Array,
  ) -> jax.Array:
    """Compute the gradient of the shape functions at the given gauss point.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinate.
      node_coords: Array of (4, 2) containing the x, and y coordinates of the nodes.

    Returns: An array of shape (4, 2) containing the derivatives of the shape
      functions with respect to the physical coordinates. The 4 corresponds to the
      number of nodes in the element and 2 corresponds to the number of physical
      dimensions.
    """
    # num_(n)odes, (d)[i]m
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)  # {nd}
    jac, _ = cls.compute_jacobian_and_determinant(gauss_pt, node_coords)
    return jnp.einsum("di, nd -> ni", jnp.linalg.inv(jac), gradN_isoparam)


class Rect4(Quad4):
  """Rectangular element with 4 nodes.

  This class is a specific instance of the Quad4 element where the element is a
  rectangle. This is useful when we want to distinguish between a general Quad4
  element and a rectangle. The element is intended to be used with a structured
  grid mesh. While the Quad4 element can be used for both structured and
  unstructured grids, the Rect4 element is specifically for structured grids. Some of
  the computations can be optimized for a structured grid.


  The nodes are numbered as follows:

        y                            3---c---2
          |                          |       |
          |                        d |       |b
          ---->x                     0---a---1
  """

  @staticmethod
  def elem_volume(node_coords: jax.Array) -> jax.Array:
    """Volume (area for quad in 2D; length in 1D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume (area) of the element.
    """
    dx = jnp.abs(node_coords[1, 0] - node_coords[0, 0])
    dy = jnp.abs(node_coords[3, 1] - node_coords[0, 1])
    return dx * dy

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the isoparametric coordinate of a point in the element.

    NOTE: No checks are performed to ensure that the point is within the element.

    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1].
    """
    dx = node_coords[1, 0] - node_coords[0, 0]
    dy = node_coords[3, 1] - node_coords[0, 1]
    xi = (2 * (point_coordn[0] - node_coords[0, 0]) / dx) - 1.0
    eta = (2 * (point_coordn[1] - node_coords[0, 1]) / dy) - 1.0
    return jnp.array([xi, eta])


def _tetrahedron_volume(tet_nodes: jax.Array) -> jax.Array:
  """Calculates the volume of a single tetrahedron.

  Args:
      tet_nodes: Array of shape (4, 3) with the coordinates of the four nodes.
  """
  # Use the formula: Volume = (1/6) * |(a-d) dot ((b-d) cross (c-d))|
  # where a, b, c, and d are the four vertices.

  a = tet_nodes[0, :]
  b = tet_nodes[1, :]
  c = tet_nodes[2, :]
  d = tet_nodes[3, :]

  volume = (1.0 / 6.0) * jnp.abs(jnp.dot(a - d, jnp.cross(b - d, c - d)))
  return volume


class Hex8:
  """Hexahedral element with 8 nodes.

  The nodes are numbered as follows:
                             7-------6
                            /|      /|
    z   y                  4-------5 |
    | /                    | 3-----| 2
    |/                     |/      |/
    0--------x             0-------1
  The edges are numbered as follows: (bottom to top, front to back, acyclic)
    0: 0-1, 1: 1-2, 2: 2-3, 3: 3-0, (bottom face)
    4: 4-5, 5: 5-6, 6: 6-7, 7: 7-4, (top face)
    8: 0-4, 9: 1-5, 10: 2-6, 11: 3-7 (vertical edges front to back, bottom to top)

  The faces are numbered as follows: (bottom to top, front to back) so that the face
  normal points outwards.
    0: 0-3-2-1, 1: 4-5-6-7, 2: 0-1-5-4, 3: 1-2-6-5, 4: 2-3-7-6, 5: 3-0-4-7
  """

  @property
  def dimension(self) -> int:
    return 3

  @property
  def num_nodes(self) -> int:
    return 8

  @property
  def meshio_type(self) -> str:
    return "hexahedron"

  @property
  def num_edges(self) -> int:
    return 12

  @property
  def num_faces(self) -> int:
    return 6

  @property
  def num_nodes_per_face(self) -> int:
    return 4

  @staticmethod
  def elem_volume(node_coords: jax.Array) -> float:
    """Volume (area in 2D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume of the element.
    """
    tet_node_indices = jnp.array(
      [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 5, 4, 6],
        [4, 6, 7, 3],
        [1, 6, 3, 4],
        [4, 5, 6, 1],
      ]
    )
    total_volume = 0.0

    for i in range(6):
      total_volume += _tetrahedron_volume(node_coords[tet_node_indices[i]])

    return total_volume

  @property
  def edge_connectivity(self) -> jax.Array:
    """Connectivity of the edges of the element.

    Returns: An array of shape (num_edges, nodes_per_edge) containing the connectivity of
      the edges of the element. Here nodes_per_edge = 2. Hence, the first column
      contains the start node of the edge and the second column contains the end node of
      the edge.
    """
    return jnp.array(
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
      ]
    )

  @property
  def face_connectivity(self) -> jax.Array:
    """Connectivity of the faces of the element.

    Returns: An array of shape (num_faces, nodes_per_face) containing the connectivity of
      the faces of the element. Here nodes_per_face = 4. The nodes of the face are in
      counter-clockwise order.
    """
    return jnp.array(
      [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
      ]
    )

  @staticmethod
  def face_normal(face_node_coords: jax.Array) -> jax.Array:
    """Normal of the faces of the element.

    Args:
      face_node_coords: Array of shape (nodes_per_face, num_dim) containing the
       coordinates of the nodes of the face. In this instance, the nodes_per_face = 4
       and num_dim is 3.

    Returns: An array of shape (num_dim,) containing the normal of a face.
    """
    v1 = face_node_coords[1, :] - face_node_coords[0, :]
    v2 = face_node_coords[3, :] - face_node_coords[0, :]
    normal = jnp.cross(v1, v2)
    normal /= jnp.linalg.norm(normal)
    return normal

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the isoparametric coordinate of a point in the element.
    NOTE: No checks are performed to ensure that the point is within the element.
    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1].
    """
    raise NotImplementedError("Not implemented for Hex8 element.")

  @staticmethod
  def shape_functions(
    isoparam_pt: jax.Array,
  ) -> jax.Array:
    """Compute the shape functions at the given xi, eta, zeta.

    Args:
      isoparam_pt: Array of (3,) of the isoparametric xi, eta, zeta coordinate.

    Returns: Array of shape (8,) containing the shape functions at the given
      xi, eta, zeta. The 8 corresponds to the number of nodes in the element.
    """

    return jnp.array(
      [
        0.125 * (1 - isoparam_pt[0]) * (1 - isoparam_pt[1]) * (1 - isoparam_pt[2]),
        0.125 * (1 + isoparam_pt[0]) * (1 - isoparam_pt[1]) * (1 - isoparam_pt[2]),
        0.125 * (1 + isoparam_pt[0]) * (1 + isoparam_pt[1]) * (1 - isoparam_pt[2]),
        0.125 * (1 - isoparam_pt[0]) * (1 + isoparam_pt[1]) * (1 - isoparam_pt[2]),
        0.125 * (1 - isoparam_pt[0]) * (1 - isoparam_pt[1]) * (1 + isoparam_pt[2]),
        0.125 * (1 + isoparam_pt[0]) * (1 - isoparam_pt[1]) * (1 + isoparam_pt[2]),
        0.125 * (1 + isoparam_pt[0]) * (1 + isoparam_pt[1]) * (1 + isoparam_pt[2]),
        0.125 * (1 - isoparam_pt[0]) * (1 + isoparam_pt[1]) * (1 + isoparam_pt[2]),
      ]
    )

  @staticmethod
  def get_physical_coordinate(
    isoparam_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the physical coordinate of a point in the element.

    Args:
      isoparam_coordn: Array of shape (num_dim,) containing the isoparametric
        coordinates of the point. The values are in the range [-1, 1].
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the physical coordinate of the
      point.
    """
    shp_fn = Hex8.shape_functions(isoparam_coordn)
    return jnp.einsum("n, nd -> d", shp_fn, node_coords)

  @staticmethod
  def shape_function_gradients_isoparametric(
    gauss_pt: jax.Array,
  ) -> jax.Array:
    """Compute the shape function gradients at the given xi, eta, zeta.

    Args:
      gauss_pt: Array of (3,) containing the isoparametric xi, eta, zeta coordinate.

    Returns: Array of (8, 3) arrays containing the shape function gradients at
      given xi, eta, zeta. The 8 corresponds to the number of nodes in the element
      and 3 corresponds to the number of physical dimensions.
    """
    xis = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1])
    etas = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zetas = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1])
    dN_dxi = jnp.array(
      [
        0.125 * xis[0] * (1 - gauss_pt[1]) * (1 - gauss_pt[2]),
        0.125 * xis[1] * (1 - gauss_pt[1]) * (1 - gauss_pt[2]),
        0.125 * xis[2] * (1 + gauss_pt[1]) * (1 - gauss_pt[2]),
        0.125 * xis[3] * (1 + gauss_pt[1]) * (1 - gauss_pt[2]),
        0.125 * xis[4] * (1 - gauss_pt[1]) * (1 + gauss_pt[2]),
        0.125 * xis[5] * (1 - gauss_pt[1]) * (1 + gauss_pt[2]),
        0.125 * xis[6] * (1 + gauss_pt[1]) * (1 + gauss_pt[2]),
        0.125 * xis[7] * (1 + gauss_pt[1]) * (1 + gauss_pt[2]),
      ]
    )
    dN_deta = jnp.array(
      [
        0.125 * (1 - gauss_pt[0]) * etas[0] * (1 - gauss_pt[2]),
        0.125 * (1 + gauss_pt[0]) * etas[1] * (1 - gauss_pt[2]),
        0.125 * (1 + gauss_pt[0]) * etas[2] * (1 - gauss_pt[2]),
        0.125 * (1 - gauss_pt[0]) * etas[3] * (1 - gauss_pt[2]),
        0.125 * (1 - gauss_pt[0]) * etas[4] * (1 + gauss_pt[2]),
        0.125 * (1 + gauss_pt[0]) * etas[5] * (1 + gauss_pt[2]),
        0.125 * (1 + gauss_pt[0]) * etas[6] * (1 + gauss_pt[2]),
        0.125 * (1 - gauss_pt[0]) * etas[7] * (1 + gauss_pt[2]),
      ]
    )
    dN_dzeta = jnp.array(
      [
        0.125 * (1 - gauss_pt[0]) * (1 - gauss_pt[1]) * zetas[0],
        0.125 * (1 + gauss_pt[0]) * (1 - gauss_pt[1]) * zetas[1],
        0.125 * (1 + gauss_pt[0]) * (1 + gauss_pt[1]) * zetas[2],
        0.125 * (1 - gauss_pt[0]) * (1 + gauss_pt[1]) * zetas[3],
        0.125 * (1 - gauss_pt[0]) * (1 - gauss_pt[1]) * zetas[4],
        0.125 * (1 + gauss_pt[0]) * (1 - gauss_pt[1]) * zetas[5],
        0.125 * (1 + gauss_pt[0]) * (1 + gauss_pt[1]) * zetas[6],
        0.125 * (1 - gauss_pt[0]) * (1 + gauss_pt[1]) * zetas[7],
      ]
    )

    return jnp.stack((dN_dxi, dN_deta, dN_dzeta), axis=1)

  @classmethod
  def compute_jacobian_and_determinant(
    cls,
    gauss_pt: jax.Array,
    node_coords: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Compute the Jacobian of the element at the given xi, eta, zeta.

    Args:
      gauss_pt: Array of (3,) containing the isoparametric xi, eta, zeta coordinate.
      node_coords: Array of (8, 3) containing the x, y, z coordinates of the nodes.

    Returns: A tuple containing the Jacobian and the its determinant:
      - Jacobians of shape (3, 3) containing the Jacobian of the element at
          the given xi, eta, zeta. 3 corresponds to the number of physical dims.
      - Determinant of shape (1,) containing the determinant of the Jacobian.
    """
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)
    # num_(n)odes, (d)[i]m
    jac = jnp.einsum("nd, ni -> di", gradN_isoparam, node_coords)
    det_jac = jnp.linalg.det(jac)
    return jac, det_jac

  def get_gradient_shape_function_physical(
    cls,
    gauss_pt: jax.Array,
    node_coords: jax.Array,
  ) -> jax.Array:
    """Compute the gradient of the shape functions at the given gauss point.

    Args:
      gauss_pt: Array of (3,) containing the isoparametric xi, eta and zeta coordinate.
      xyz_nodes: Array of (8, 3) containing the x, y and z coordinates of the nodes.

    Returns: An array of shape (8, 3) containing the derivatives of the shape
      functions with respect to the physical coordinates. The 8 corresponds to the
      number of nodes in the element and 3 corresponds to the number of physical
      dimensions.
    """
    # (g)auss points, num_(n)odes, (d)[i]m
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)  # {nd}
    jac, _ = cls.compute_jacobian_and_determinant(gauss_pt, node_coords)
    return jnp.einsum("di, nd -> ni", jnp.linalg.inv(jac), gradN_isoparam)


class Cube8(Hex8):
  """Cuboidal element with 8 nodes.

  This class is a specific instance of the Hex8 element where the element is a
  cube. This is useful when we want to distinguish between a general Hex8
  element and a cube. The element is intended to be used with a structured
  grid mesh. While the Hex8 element can be used for both structured and
  unstructured grids, the Cube8 element is specifically for structured grids. Some of
  the computations can be optimized for a structured grid.

    The nodes are numbered as follows:
                             7-------6
                            /|      /|
    z   y                  4-------5 |
    | /                    | 3-----| 2
    |/                     |/      |/
    0--------x             0-------1
  """

  @staticmethod
  def elem_volume(node_coords: jax.Array) -> jax.Array:
    """Volume (area for quad in 2D; length in 1D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume (area) of the element.
    """
    dx = jnp.abs(node_coords[1, 0] - node_coords[0, 0])
    dy = jnp.abs(node_coords[3, 1] - node_coords[0, 1])
    dz = jnp.abs(node_coords[4, 2] - node_coords[0, 2])
    return dx * dy * dz

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: jax.Array, node_coords: jax.Array
  ) -> jax.Array:
    """Get the isoparametric coordinate of a point in the element.

    NOTE: No checks are performed to ensure that the point is within the element.

    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1] (no checks done).
    """
    dx = node_coords[1, 0] - node_coords[0, 0]
    dy = node_coords[3, 1] - node_coords[0, 1]
    dz = node_coords[4, 2] - node_coords[0, 2]
    xi = (2 * (point_coordn[0] - node_coords[0, 0]) / dx) - 1.0
    eta = (2 * (point_coordn[1] - node_coords[0, 1]) / dy) - 1.0
    zeta = (2 * (point_coordn[2] - node_coords[0, 2]) / dz) - 1.0
    return jnp.array([xi, eta, zeta])
"""Structural coupling for pressure in fluid-structure interaction (FSI) problems."""

import jax
import jax.numpy as jnp

import toflux.src.mesher as _mesh


def _compute_elem_pressure_force(
  pressure_coupling_filter: jax.Array,
  elem_pressure: jax.Array,
  node_coords: jax.Array,
  shp_fn: jax.Array,
  mesh: _mesh.Mesh,
):
  """Compute the pressure force on an element.

  Args:
    pressure_coupling_filter: Scalar array containing the pressure coupling filter
      for the element.
    elem_pressure: Array of (num_nodes_per_elem,) containing the pressure at the nodes
      of the element.
    node_coords: Array of (num_nodes_per_elem, num_dim) containing the coordinates of
      the nodes of the element.
    shp_fn: Array of (num_gauss_pts, num_nodes_per_elem) containing the shape
      functions evaluated at the Gauss points.
    mesh: The structural mesh object.
  """
  # (d)(i)m, (g)auss, (n)odes_per_elem

  # first coupling term: pressure gradient
  grad_shp_fn = jax.vmap(
    mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
  )(mesh.gauss_pts, node_coords)  # (g, n, d)
  _, det_jac = jax.vmap(
    mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
  )(mesh.gauss_pts, node_coords)  # (g,)

  dpress_xy = jnp.einsum("gnd, n -> gd", grad_shp_fn, elem_pressure)

  p_coupl_1 = pressure_coupling_filter * jnp.einsum(
    "gn, gd, g, g -> nd", shp_fn, dpress_xy, det_jac, mesh.gauss_weights
  )

  # second coupling term: pressure times divergence of shape function
  press_gauss = jnp.einsum("gn, n -> g", shp_fn, elem_pressure)
  p_coupl_2 = pressure_coupling_filter * jnp.einsum(
    "gnd, g, g, g -> nd", grad_shp_fn, press_gauss, det_jac, mesh.gauss_weights
  )
  return (p_coupl_1 + p_coupl_2).ravel()


def compute_pressure_force(
  pressure: jax.Array,
  struct_mesh: _mesh.Mesh,
  pressure_coupling_filter: jax.Array,
):
  """Compute the force due to pressure on the structural mesh.

  This function computes the force on the structure due to the pressure from the fluid.
    The pressure is applied to the nodes of the structural mesh, and the forces are
    computed.

    The weak form of the structural equilibrium equation with pressure coupling is:

    ∫_Ω (∂w_i / ∂x_j) σ_ij(ρ) dV
    = ∫_Ω Ψ(ρ) w_i (∂p / ∂x_i) dV
    + ∫_Ω Ψ(ρ) (∂w_i / ∂x_i) p dV
    + ∫_Ω w_i f_i dV

    First term (left-hand side):  The structural stiffness term in the weak form:
    ∫_Ω (∂w_i / ∂x_j) σ_ij(ρ) dV
    where:
    - w_i^h is the structural test (shape) function.
    - σ_ij^(ρ) is the structural stress tensor, dependent on the density distribution ρ.

    Second term: The first pressure coupling term, describing the effect of fluid
      pressure gradients on the structure:
    ∫_Ω Ψ(ρ) w_i (∂p / ∂x_i) dV
    where:
    - Ψ(ρ) is a density-dependent interpolation function controlling pressure coupling.
    - p is the fluid pressure field.

    Third term: The second pressure coupling term, describing the structural volume
      deformation due to fluid pressure:
    ∫_Ω Ψ(ρ) (∂w_i / ∂x_i) pdV
    where:
    - (∂w_i / ∂x_i) represents the divergence of the shape function.
    - p is the fluid pressure field.

    Fourth term: The external structural body forces applied directly to the solid:
    ∫_Ω w_i f_i dV
    where:
    - f_i represents external forces acting on the structure.
    - w_i is the shape function.

  For more details see eq(3) in :
    Lundgaard, Christian, etal. "Revisiting density-based topology optimization for
    fluid-structure-interaction problems." SMO 58, no. 3 (2018): 969-995.

  Args:
    pressure: Array of (num_nodes,) containing the pressure at the nodes of the
      structural mesh.
    struct_mesh: Mesh object of the structural mesh.
    pressure_coupling_filter: Array of (num_elems,) containing the pressure coupling
      filter. This is a mask that indicates which elements of are sold and which are
      fluid. Typically, this is a binary mask where 1 indicates a solid element and 0
      indicates a fluid element.

  Returns: An array of (num_elems, num_dofs_per_elem) containing the force due to
    pressure on the structure. The forces are arranged in the order of [f_1_x, f_1_y,
    f_1_z, f_2_x, f_2_y, ...] where f_i_x, f_i_y, f_i_z are the forces on the
    i-th node of the element.
  """
  # (d)(i)m, (g)auss, (n)odes_per_elem
  shp_fn = jax.vmap(struct_mesh.elem_template.shape_functions)(
    struct_mesh.gauss_pts
  )  # (g, n)
  return jax.vmap(_compute_elem_pressure_force, in_axes=(0, 0, 0, None, None))(
    pressure_coupling_filter,
    pressure[struct_mesh.elem_nodes],
    struct_mesh.elem_node_coords,
    shp_fn,
    struct_mesh,
  )

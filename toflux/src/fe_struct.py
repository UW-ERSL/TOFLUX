"""Structural solver with large deformations.

This module implements a finite element solver for solid mechanics problems,
capable of handling both small (linear) and large (geometrically non-linear)
deformations. The solver is based on a total Lagrangian formulation.

The governing equation is the balance of linear momentum in its strong form
(for a static problem):

    ∇ ⋅ P + B = 0

where:
  P : First Piola-Kirchhoff stress tensor.
  B : Body force vector in the reference configuration.

The weak form of this equation states that the internal virtual work must equal
the external virtual work for any admissible virtual displacement. This leads to
the residual equation for a single element:

    R_e = F_int - F_ext = 0

where:
  F_int : Nodal internal forces, computed from the stress state within the element.
  F_ext : Nodal external forces (applied loads, body forces, etc.).
"""

import enum
import numpy as np
import jax.numpy as jnp
import jax
import jax.experimental.sparse as jax_sprs
from jax.typing import ArrayLike

import toflux.src.solver as _nlsolv
import toflux.src.mesher as _mesh
import toflux.src.material as _mat
import toflux.src.bc as _bc



class DisplacementField(enum.Enum):
  """The displacment fields."""

  U = 0
  V = 1
  W = 2


class DeformationModel(enum.Enum):
  SMALL = enum.auto()
  LARGE = enum.auto()


def _compute_deformation_gradient_from_displacement(
  u: jnp.ndarray,
  grad_shape_fn: jnp.ndarray,
) -> jnp.ndarray:
  """Get the deformation gradient matrix.

  The deformation gradient matrix is defined as:

          F_ij = delta_ij + grad_j(u_i)

    Where grad_j(u_i) is evaluated as sum_i(grad_shape_fn[i] * u[i]); summing over all
    the shape functions. This is from the observation that the displacement at a point
    is approximated as a linear combination of the shape functions. Then, the gradient
    of the displacement is approximated as a linear combination of the gradient of the
    shape functions.

  Args:
    u: Array of (num_nodes, num_dim) containing the displacement at a node.
    grad_shape_fn: Array of (num_shape_fn, num_dim) containing the gradient of the
      shape functions.
  Returns:
    A array of (num_dim, num_dim) containing the deformation gradient matrix.
  """
  # d(i)(m), (s)hape_fn
  _, num_dim = u.shape
  identity = jnp.eye(num_dim)
  return jnp.einsum("si, sm -> im", grad_shape_fn, u) + identity


class FEA(_nlsolv.NonlinearProblem):
  """Structural finite element analysis accounting for large deformations."""

  def __init__(
    self,
    mesh: _mesh.Mesh,
    material: _mat.StructuralMaterial,
    deformation_model: DeformationModel,
    bc: _bc.BCDict,
    solver_settings: dict,
  ):
    """Initializes the FEA problem."""
    super().__init__(solver_settings=solver_settings)
    self.mesh, self.material, self.bc = mesh, material, bc
    self.deformation_model = deformation_model

  def compute_elem_residual(
    self,
    u_elem: ArrayLike,
    lame_lambda: ArrayLike,
    lame_mu: ArrayLike,
    force_elem: ArrayLike,
    node_coords: ArrayLike,
  ) -> ArrayLike:
    """Compute the residual of an element.

    This function computes the residual of an element using the finite element
    method. The residual is the difference between the internal forces and the
    external forces applied to the element.

                          R = F_ext - F_int

    Args:
      u_elem: Array of (num_dofs_per_elem,) containing the displacements of the nodes of
        an element. The displacements are assumed to be ordered as (u1, v1, u2, v2, ...).
      lame_lambda: Scalar value of the Lame (first) parameter lambda.
      lame_mu: Scalar value of the Lame (second) parameter mu.
      force_elem: Array of (num_dofs_per_elem,) containing the external forces applied to
        the element nodes. The forces are assumed to be ordered as (fx1, fy1, fx2, ...).
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.

    Returns: The residual of the element (num_dofs_per_elem,).
    """
    # (g)auss, (d)(i)(m), (n)odes_per_elem
    u_el = u_elem.reshape(
      (self.mesh.elem_template.num_nodes, self.mesh.num_dim)
    )  # (n, d)

    grad_shp_fn_physical = jax.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g, n, d)
    _, det_jac = jax.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,)

    def_grad = jax.vmap(
      _compute_deformation_gradient_from_displacement, in_axes=(None, 0)
    )(u_el, grad_shp_fn_physical)  # (g,d,d)

    pk2_stress = jax.vmap(_mat.compute_hookean_pk2, in_axes=(0, None))(
      def_grad, (lame_lambda, lame_mu)
    )  # (g, d, d)
    bjw = jnp.einsum(
      "g, g, gnd -> gnd", det_jac, self.mesh.gauss_weights, grad_shp_fn_physical
    )

    if self.deformation_model == DeformationModel.SMALL:
      stress = pk2_stress
    elif self.deformation_model == DeformationModel.LARGE:
      stress = jnp.einsum("gdi, gim -> gdm", def_grad, pk2_stress)  # pk1

    force_internal = jnp.einsum("gdi, gnd -> ni", stress, bjw).flatten()
    return force_elem - force_internal

  def get_residual_and_tangent_stiffness(
    self,
    u: ArrayLike,
    lame_lambda: ArrayLike,
    lame_mu: ArrayLike,
    addn_force: ArrayLike = None,
  ) -> tuple[ArrayLike, jax_sprs.BCOO]:
    """Compute the residual and tangent matrix of the system of equations.
        
    The residual is given by:

                  R = F_ext - F_int

      where F_ext is the external force vector and F_int is the internal force vector
      computed from the element stresses.

      The tangent stiffness matrix is computed as the derivative of the residual
      with respect to the displacements: 
                  K = dR/du
      
      In our framework, we compute the tangent stiffness matrix automatically using
      JAX's automatic differentiation capabilities.

    NOTE: This function overrides the base class method to provide the specific
      implementation for structural mechanics problems. Similar adaptation follows for
      the fluid and thermal solvers.

    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
      lame_lambda: Array of size (num_elems,) that contain the Lame (first) parameter
        lambda for each element.
      lame_mu: Array of size (num_elems,) that contain the Lame (second) parameter mu for
        each element.
      addn_force: Optional array of size (num_elems, num_dofs_per_elem) that contains
        additional forces to be added to the element forces. This is useful for applying
        external forces such as from fluid pressure/thermal loads.

    Returns: Array of size (num_dofs,) which is the residual of the system of
      equations.
    """
    # (e)lement, (d)ofs_per_elem
    u_elem = u[self.mesh.elem_dof_mat]  # {ed}
    elem_forces =  self.bc["elem_forces"]
    if addn_force is not None:
      elem_forces = self.bc["elem_forces"] + addn_force

    res_args = (u_elem, lame_lambda, lame_mu, elem_forces, self.mesh.elem_node_coords)
    # residual
    elem_residual = jax.vmap(self.compute_elem_residual)(*res_args)  # {ed}
    residual = jnp.zeros((self.mesh.num_dofs,))
    residual = residual.at[self.mesh.elem_dof_mat].add(elem_residual)
    residual = residual.at[self.bc["fixed_dofs"]].set(0.0)

    # tangent stiffness
    elem_jac = jax.vmap(jax.jacfwd(self.compute_elem_residual, argnums=0))(*res_args)
    node_idx = np.stack((self.mesh.iK, self.mesh.jK)).astype(np.int32).T
    assm_jac = jax_sprs.BCOO(
      (elem_jac.flatten(order="C"), node_idx),
      shape=(self.mesh.num_dofs, self.mesh.num_dofs),
    )
    assm_jac = _bc.apply_dirichlet_bc(assm_jac, self.bc["fixed_dofs"])
    return residual, assm_jac

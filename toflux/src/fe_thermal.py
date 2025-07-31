"""Conjugate heat transfer solver.


DOCUMENTATION NEEDED




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


class ThermalField(enum.Enum):
  """The temperature field."""

  TEMPERATURE = 0


class FEA(_nlsolv.NonlinearProblem):
  """Conjugate heat transfer finite element analysis with advection and diffusion."""

  def __init__(
    self,
    mesh: _mesh.Mesh,
    material: _mat.ThermalMaterial,
    bc: _bc.BCDict,
    solver_settings: dict,
  ):
    """Initializes the conjugate heat trasfer FEA problem."""
    super().__init__(solver_settings=solver_settings)
    self.mesh, self.material, self.bc = (
      mesh,
      material,
      bc,
    )
    self.node_id_jac = np.stack((self.mesh.iK, self.mesh.jK)).astype(np.int32).T
    self.shp_fn = jax.vmap(self.mesh.elem_template.shape_functions)(mesh.gauss_pts)

  def _compute_elem_stabilization(
    self,
    velocity: jnp.ndarray,
    diffusivity: jnp.ndarray,
    elem_char_length: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the stabilization parameter for SUPG term.

      This function computes the stabilization parameter (τ) used in
        convection-diffusion problems for each element.The stabilization
        parameter (τ) is computed using an approximate minimum function considering
        two limiting cases:

      - τ₁: Convective limit
      - τ₃: Diffusive limit

      The stabilization parameter is assumed constant within each element, and
      τ₁ is computed based on the velocity components evaluated at the element centroid.

      Stabilization parameter (τ) is computed as:

      τ = ( τ₁⁻² + τ₃⁻² )^(-1/2)

      Where:
      τ₁ = h / ( 2√(uᵢ uᵢ) )   # Convective limit
      τ₃ = h² / (12 α)         # Diffusive limit
      τ_2 is the transient limit and ignored

      Variables:
      - h: Element characteristic length
      - uᵢ: Velocity components
      - α: Thermal diffusivity

     For details see [Alexandreasen 2023 SMO], Appendix B, eq 49
     https://link.springer.com/article/10.1007/s00158-022-03420-9

    Args:
      velocity: An array of shape (num_velocity_dofs_per_elem,) of the velocity field at
        the element dofs. The velocity is assumed to be in the order of
        [u₁, v₁, w₁, u₂, v₂, w₂, ...] for 2D and 3D problems.
      diffusivity: A scalar array containing the thermal diffusivity.
      elem_char_length: A scalar array containing the characteristic length of the
        element.

    Returns: The stabilization parameter.
    """
    gp_center = jnp.zeros((self.mesh.num_dim,))
    shp_fn = self.mesh.elem_template.shape_functions(gp_center)

    # (d)(i)m, (g)auss, (n)odes_per_elem
    u0 = jnp.einsum("n, nd -> nd", shp_fn, velocity.reshape(-1, self.mesh.num_dim))
    ue = jnp.einsum("nd, nd -> ", u0, u0)

    inv_sq_tau1 = (4 * ue) / elem_char_length**2
    tau_3 = (elem_char_length**2) / (12 * diffusivity)

    return (inv_sq_tau1 + tau_3 ** (-2)) ** (-1 / 2)

  def _compute_elem_residual(
    self,
    temperature: jnp.ndarray,
    velocity: jnp.ndarray,
    diffusivity: jnp.ndarray,
    node_coords: jnp.ndarray,
    elem_char_length: float,
  ) -> jnp.ndarray:
    """Computes the elemental residual of the thermal stiffness matrix.

    The weak form of the steady-state energy equation (dimensional form) with SUPG
    stabilisation can be written as:

        ∫_Ω_e  w * u_j * ∂T/∂x_j dΩ_e                   (convection)
      + ∫_Ω_e  (∂w/∂x_j) * α * ∂T/∂x_j  dΩ_e            (diffusion)
      + ∫_Ω_e  τ_T * u_j * ∂w/∂x_j * R_T(u, T) dΩ_e    (SUPG)
      = 0

    where:
          Ω_e      : entire analysis domain
          u_j      : velocity component in direction x_j
          T        : temperature field
          w        : weight / test function
          α        : thermal diffusivity
          τ_T      : SUPG stabilisation parameter
          R_T(u,T) : strong-form residual of the energy equation

    Where:
          R_T = u_j * (∂T/∂x_j)  - α* (∂²T/∂x_j∂x_j)


    For more details
      1. see eq (2d) in:
        Subramaniam, V.,etal. "Topology optimization of conjugate heat transfer systems:
        A competition between heat transfer enhancement and pressure drop reduction."
        International Journal of Heat and Fluid Flow 75 (2019): 165-184.
      2. eq (A.44) in:
        Alexandersen, Joe "Topology optimisation for coupled convection problems" (2013)

    NOTE: This implementation assumes there are no externally applied surface heat flux
      or volumetric heat sources. This simplification is valid only for the opimization
      problems considered herein. For problems such as heat sinks, with heat generation
      these terms need to be added to the residual.

    Args:
      temperature: Array of (num_dofs_per_elem,) containing the temperature of the nodes
        of an element.
      velocity: Array of (num_nodes_per_elem * num_dim,) containing the velocity at the
        nodes of an element. The velocity  are assumed to be ordered as
        (u1, v1, w2 u2, v2, w2...) etc. The velocity is part of the convective heat
        transfer.
      diffusivity: Scalar value of the element's thermal diffusivity.
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.
      elem_char_length: Scalar value of the diagonal length of the element.

    Returns: Array of (num_dofs_per_elem,) containing the residual of the element. The
      resiudal's ordered is assumed as (t1, t2, t3,...) of the temperature at the nodes.
    """
    # (d)(i)m, (g)auss, (n)odes_per_elem = (t)emp_dofs_per_elem, (v)el_dofs_per_elem
    grad_shp_fn = jax.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,n,d)

    _, det_jac = jax.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)

    stab_param = self._compute_elem_stabilization(
      velocity, diffusivity, elem_char_length
    )
    vel_gauss = jnp.einsum(
      "gn, nd -> gd", self.shp_fn, velocity.reshape(-1, self.mesh.num_dim)
    )
    dtemp_xy = jnp.einsum("gnd, n -> gd", grad_shp_fn, temperature)

    res_conv = jnp.einsum("gn, gd, gd -> gn", self.shp_fn, vel_gauss, dtemp_xy)
    res_diff = diffusivity * jnp.einsum("gnd, gd -> gn", grad_shp_fn, dtemp_xy)

    res_strong_form = jnp.einsum("gd, gd -> g", vel_gauss, dtemp_xy)
    u_dot_grad_w = jnp.einsum("gd, gnd -> gn", vel_gauss, grad_shp_fn)
    res_conv_supg = stab_param * jnp.einsum("gn, g-> gn", u_dot_grad_w, res_strong_form)

    net_res = res_diff + res_conv + res_conv_supg
    return jnp.einsum("gn, g, g -> n", net_res, self.mesh.gauss_weights, det_jac)

  def get_residual_and_tangent_stiffness(
    self,
    temperature: ArrayLike,
    elem_velocity: ArrayLike,
    diffusivity: ArrayLike,
  ) -> ArrayLike:
    """Compute the residual of the system of equations.
        The residual is given by:

                  R = K u - f

    Args:
      temp: Array of size (num_dofs,) which is the temperature of the nodes of the mesh.
      elem_velocity: Array of size (num_elems, num_nodes_per_elem*num_dim) that contain
        the velocity at the nodes of the elements. The velocity is assumed to be ordered
        as (u1, v1, w1, u2, v2, w2, ...).
      diffusivity: Array of size (num_elems,) that contain the thermal diffusivity of the
        elements.

    Returns:
      residual: Array of size (num_dofs,) which is the residual of the system.
      assm_jac: Sparse matrix of size (num_dofs, num_dofs) which is the tangent stiffness
        matrix of the system.
    """
    # (e)lement, (d)ofs_per_elem
    temp_elem = temperature[self.mesh.elem_dof_mat]  # {ed}

    res_args = (
      temp_elem,
      elem_velocity,
      diffusivity,
      self.mesh.elem_node_coords,
      self.mesh.elem_diag_length,
    )
    # residual
    elem_residual = jax.vmap(self._compute_elem_residual)(*res_args)  # {ed}
    residual = jnp.zeros((self.mesh.num_dofs,))
    residual = residual.at[self.mesh.elem_dof_mat].add(elem_residual)
    residual = residual.at[self.bc["fixed_dofs"]].set(0.0)

    # tangent stiffness
    elem_jac = jax.vmap(jax.jacfwd(self._compute_elem_residual, argnums=0))(*res_args)
    assm_jac = jax_sprs.BCOO(
      (elem_jac.flatten(), self.node_id_jac),
      shape=(self.mesh.num_dofs, self.mesh.num_dofs),
    ).T
    assm_jac = _bc.apply_dirichlet_bc(assm_jac, self.bc["fixed_dofs"])
    return residual, assm_jac

"""Fluid flow solver for Navier-Stokes equations in dimensional form."""

import enum
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
from jax.typing import ArrayLike

import toflux.src.mesher as _mesher
import toflux.src.material as material
import toflux.src.bc as _bc
import toflux.src.solver as _solv


class FluidField(enum.Enum):
  """The pressure-velocity fields."""

  PRESSURE = 0
  U_VEL = 1
  V_VEL = 2


class FluidSolver(_solv.NonlinearProblem):
  """Fluid flow solver for Navier-Stokes equations."""

  def __init__(
    self,
    mesh: _mesher.Mesh,
    bc: _bc.BCDict,
    material: material.FluidMaterial,
    solver_settings: dict,
  ):
    super().__init__(solver_settings=solver_settings)
    self.bc = bc
    self.material = material
    self.mesh = mesh
    self.node_id_jac = np.stack((self.mesh.iK, self.mesh.jK)).astype(np.int32).T
    self.shp_fn = jax.vmap(self.mesh.elem_template.shape_functions)(mesh.gauss_pts)

  def _compute_elem_stabilization(
    self,
    velocity: jnp.ndarray,
    brinkman_penalty: jnp.ndarray,
    elem_char_length: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the stabilization parameter for SUPG term.

      This function computes the stabilization parameter (τ) used in
        convection-diffusion-reaction problems computed for each element.
      The stabilization parameter (τ) is computed using an approximate minimum
        function considering three limiting cases:

      - τ₁: Convective limit
      - τ₃: Diffusive limit
      - τ₄: Reactive limit

      The reactive limit (τ₄) is especially important for ensuring stability in
      the solid domain and at the interface, particularly for large Brinkman penalty
      parameters.
      The stabilization parameter is assumed constant within each element, and
      τ₁ is computed based on the velocity components evaluated at the element centroid.

      Stabilization parameter (τ) is computed as:

      τ = ( τ₁⁻² + τ₃⁻² + τ₄⁻² )^(-1/2)

      Where:
      τ₁ = h / ( 2√(uᵢ uᵢ) )   # Convective limit
      τ₃ = ρh² / (12μ)         # Diffusive limit
      τ₄ = ρ / α              # Reactive limit
      τ_2 is the transient limit and ignored

      Variables:
      - h: Element characteristic length
      - uᵢ: Velocity components
      - ρ: Density
      - μ: Dynamic viscosity
      - α: Brinkman penalty parameter (used for solid domain stabilization)

     For details see [Alexandreasen 2023 SMO], Appendix B, eq 49
     https://link.springer.com/article/10.1007/s00158-022-03420-9

    Args:
      velocity: An array of shape (num_velocity_dofs_per_elem,) of the velocity field at
        the element dofs.
      brinkman_penalty: A scalar array containing the Brinkman penalty.
      elem_char_length: A scalar array containing the characteristic length of the
        element.

    Returns:
      stab_param: Stabilization parameter.
    """
    gp_center = jnp.zeros((self.mesh.num_dim,))
    shp_fn = self.mesh.elem_template.shape_functions(gp_center)

    # (d)(i)m, (g)auss, (n)odes_per_elem
    u0 = jnp.einsum("n, nd -> nd", shp_fn, velocity.reshape(-1, self.mesh.num_dim))
    ue = jnp.einsum("nd, nd -> ", u0, u0)

    inv_sq_tau1 = (4 * ue) / elem_char_length**2
    tau_3 = (self.material.mass_density * elem_char_length**2) / (
      12 * self.material.dynamic_viscosity
    )
    tau_4 = self.material.mass_density / brinkman_penalty

    stab_param = (inv_sq_tau1 + tau_3 ** (-2) + tau_4 ** (-2)) ** (-1 / 2)
    return stab_param

  def _compute_elem_residual(
    self,
    pressure_velocity: jnp.ndarray,
    brinkman_penalty: jnp.ndarray,
    node_coords: jnp.ndarray,
    elem_char_length: float,
  ) -> jnp.ndarray:
    """Computes the elemntal residual of the fluid stiffness matrix.

    The weak form is expressed as follows:

    Integral Terms:
    ----------------
    First term: Represents the convection term in the weak form:
      ∫_Ω ρ w_i (u_j ∂u_i / ∂x_j) dV
      where ρ is the mass density, w_i is the shape function, and u_i is the velocity.

    Second term: Represents the viscous diffusion term:
      ∫_Ω μ (∂w_i / ∂x_j) (∂u_i / ∂x_j + ∂u_j / ∂x_i) dV
      where μ is the dynamic viscosity.

    Third term: Represents the pressure term:
      ∫_Ω (∂w_i / ∂x_i) p dV
      where p is the pressure field.

    Fourth term: Represents the Brinkman penalization (Darcy friction) term:
      ∫_Ω α w_i u_i dV
      where α is the inverse permeability (related to the Darcy number).

    Last terms: SUPG stabilization term:
      ∫_Ωe τ w_k (∂u_i / ∂x_k) (ρ u_j ∂u_i / ∂x_j + ∂p / ∂x_i + α u_i) dV
      where τ is the stabilization parameter to handle convection-dominated problems.

    The stabilization terms are added using Streamline Upwind Petrov-Galerkin (SUPG)
      method to enhance stability by adding artificial diffusion in the streamline
      direction, especially important for higher Reynolds number flows.

    - First term: Represents the weak form of mass conservation (continuity equation):
      ∫_Ω q (∂u_i / ∂x_i) dV
      where `q` is the test function for pressure, and `u_i` is the velocity field.

    - last three terms: PSPG stabilization term:
      ∑_e ∫_Ωe τ (ρ ∂q / ∂x_i) (u_j ∂u_i / ∂x_j + ∂p / ∂x_i + α u_i) dV
      where:
        - `τ` is the PSPG stabilization parameter.
        - `ρ` is the density.
        - `u_i` is the velocity vector.
        - `p` is the pressure field.
        - `α` is the inverse permeability (related to the Darcy number).
      This term enhances stability by adding pressure stabilization, which is
      crucial for avoiding numerical issues such as pressure oscillations in mixed
      formulations, especially in convection-dominated flows.

    For details see [Alexandreasen 2023 SMO], Appendix B, eq 47
     https://link.springer.com/article/10.1007/s00158-022-03420-9


    Args:
      pressure_velocity: Array of (num_dofs_per_elem,) containing the velocity and
        pressure of the nodes of an element. The velocity and pressure are assumed to be
        ordered as (p1, u1, v1, p2, u2, v2, ...).
      brinkman_penalty: Scalar value of the elemntwise brinkman penalty.
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.
      elem_char_length: Scalar value of the diagonal length of the element.

    Returns: Array of (num_dofs_per_elem,) containing the residual of the element. The
      residual is assumed to be ordered as (p1, u1, v1, p2, u2, v2, ...) at the nodes.
    """
    # (d)(i)m, (g)auss, (n)odes_per_elem, vel_d(o)fs_per_elem
    num_gauss_pts = self.mesh.gauss_weights.shape[0]
    num_fields = self.mesh.num_dim + 1  # velocity + pressure

    grad_shp_fn = jax.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,n,d)

    _, det_jac = jax.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)

    velocity = pressure_velocity.reshape(-1, num_fields)[:, 1:].flatten()
    pressure = pressure_velocity.reshape(-1, num_fields)[:, 0].flatten()

    stab_param = self._compute_elem_stabilization(
      velocity, brinkman_penalty, elem_char_length
    )

    vel_gauss = jnp.einsum(
      "gn, nd -> gd", self.shp_fn, velocity.reshape(-1, self.mesh.num_dim)
    )
    press_gauss = jnp.einsum("gn, n -> g", self.shp_fn, pressure)

    dvel_xy = jnp.einsum(
      "gnd, ni -> gid", grad_shp_fn, velocity.reshape(-1, self.mesh.num_dim)
    )
    dpress_xy = jnp.einsum("gnd, n -> gd", grad_shp_fn, pressure)

    # Momentum equations
    res_viscosity = self.material.dynamic_viscosity * jnp.einsum(
      "gnd, gdi -> gni", grad_shp_fn, dvel_xy + dvel_xy.transpose(0, 2, 1)
    )

    res_brink = brinkman_penalty * jnp.einsum("gn, gd -> gnd", self.shp_fn, vel_gauss)

    res_conv = self.material.mass_density * jnp.einsum(
      "gn, gd, gid -> gni", self.shp_fn, vel_gauss, dvel_xy
    )

    res_brink_stab = (
      jnp.einsum("gnd, gd, gd -> gnd", grad_shp_fn, vel_gauss, vel_gauss)
      * stab_param
      * brinkman_penalty
    )

    res_conv_stab = (
      self.material.mass_density
      * jnp.einsum(
        "gnd, gi, gm, gdm -> gni", grad_shp_fn, vel_gauss, vel_gauss, dvel_xy
      )
      * stab_param
    )

    # SUPG Pressure term
    res_press_stab = (
      jnp.einsum("gnd, gd, gi -> gni", grad_shp_fn, vel_gauss, dpress_xy) * stab_param
    )

    res_press = jnp.einsum("gnd, g -> gnd", grad_shp_fn, press_gauss)

    res_mom = (
      res_viscosity
      + res_brink
      + res_conv
      + res_brink_stab
      + res_conv_stab
      + res_press_stab
      - res_press
    ).reshape((num_gauss_pts, -1))

    # Incomressibility equations
    res_press_div = jnp.einsum("gn, gdd -> gn", self.shp_fn, dvel_xy)

    res_press_brink_stab = (
      (stab_param / self.material.mass_density)
      * brinkman_penalty
      * jnp.einsum("gnd, gd -> gn", grad_shp_fn, vel_gauss)
    )

    res_press_conv_stab = stab_param * jnp.einsum(
      "gni, gd, gid -> gn", grad_shp_fn, vel_gauss, dvel_xy
    )

    res_press_stab = (stab_param / self.material.mass_density) * jnp.einsum(
      "gnd, gd -> gn", grad_shp_fn, dpress_xy
    )

    res_incom = (
      res_press_div + res_press_brink_stab + res_press_conv_stab + res_press_stab
    )
    res_mom = jnp.einsum("go, g, g -> o", res_mom, self.mesh.gauss_weights, det_jac)
    res_incom = jnp.einsum("gn, g, g -> n", res_incom, self.mesh.gauss_weights, det_jac)

    res = jnp.column_stack([res_incom, res_mom[0::2], res_mom[1::2]]).ravel()
    return res

  def get_residual_and_tangent_stiffness(
    self,
    press_vel: ArrayLike,
    brinkman_penalty: ArrayLike,
  ) -> ArrayLike:
    """Compute the residual of the system of equations.
        The residual is given by:

                  R = K u - f
    where:
      - R is the residual vector.
      - K is the stiffness matrix.
      - u is the solution vector.
      - f is the load vector.

    Args:
      press_vel: Array of size (num_dofs,) which is the pressure and velocity of the nodes
        of the mesh.
      brinkman_penalty: Array of size (num_elems,) that contain the Brinkman penalty for
        each element.

    Returns: Array of size (num_dofs,) which is the residual of the system of
      equations.
    """
    elem_pressure_velocity_field = press_vel[self.mesh.elem_dof_mat]

    res_args = (
      elem_pressure_velocity_field,
      brinkman_penalty,
      self.mesh.elem_node_coords,
      self.mesh.elem_diag_length,
    )
    elem_residual = jax.vmap(self._compute_elem_residual)(*res_args)

    residual = jnp.zeros((self.mesh.num_dofs,))
    residual = residual.at[self.mesh.elem_dof_mat].add(elem_residual)
    residual = residual.at[self.bc["fixed_dofs"]].set(0.0)

    elem_jacobian = jax.vmap(jax.jacrev(self._compute_elem_residual, argnums=0))(
      *res_args
    )
    assembled_jacobian_matrix = jax_sprs.BCOO(
      (elem_jacobian.flatten(), self.node_id_jac),
      shape=(self.mesh.num_dofs, self.mesh.num_dofs),
    ).T
    assm_jac = _bc.apply_dirichlet_bc(assembled_jacobian_matrix, self.bc["fixed_dofs"])
    return residual, assm_jac

  def compute_elem_dissipated_power(
    self,
    brinkman_penalty: jnp.ndarray,
    pressure_velocity: jnp.ndarray,
    node_coords: jnp.ndarray,
  ) -> float:
    """Calculates the dissipated power for the fluid flow problem at given element.

    The addition of the Brinkman penalty term introduces a body force that must be
      considered in the calculation of the total dissipated energy in the system.

    The dissipated energy is expressed as:

    ϕ = (1/2) * ∫_Ω [ μ (∂u_i/∂x_j * ∂u_i/∂x_j + ∂u_j/∂x_i) + α(x) * u_i * u_i ]dV

    Where:
    --------
      ϕ: Dissipated energy in the system.
      μ: Dynamic viscosity of the fluid.
      u_i: Velocity components.
      ∂u_i / ∂x_j: Gradient of the velocity field in the x-direction.
      α(x): Brinkman penalty term, which acts as a body force and penalizes the
      velocity field within the porous domain.
      Ω: The domain over which the integration is performed.

    Args:
      brinkman_penalty: Scalar value of the elemntwise brinkman penalty.
      press_velocity : Array of (num_dofs_per_elem,) containing the velocity and
        pressure of the nodes of an element. The velocity and pressure are assumed to be
        ordered as (p1, u1, v1, p2, u2, v2, ...).
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.

    Returns: The computed objective value.
    """
    num_fields = self.mesh.num_dim + 1  # velocity + pressure
    vel = pressure_velocity.reshape(-1, num_fields)[:, 1:].flatten()
    _, det_jac = jax.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)

    # (d)(i)m, (g)auss, (n)odes_per_elem
    grad_shp_fn = jax.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,n,d)
    dvel_xy = jnp.einsum(
      "gnd, ni -> gid", grad_shp_fn, vel.reshape(-1, self.mesh.num_dim)
    )
    vel_gauss = jnp.einsum(
      "gn, nd -> gd", self.shp_fn, vel.reshape(-1, self.mesh.num_dim)
    )

    obj_integ = 0.5 * (
      brinkman_penalty * jnp.einsum("gd, gd-> g", vel_gauss, vel_gauss)
      + self.material.dynamic_viscosity
      * jnp.einsum(
        "gdi, gdi -> g", dvel_xy, dvel_xy + jnp.transpose(dvel_xy, (0, 2, 1))
      )
    )
    return jnp.einsum("g, g, g -> ", obj_integ, self.mesh.gauss_weights, det_jac)

"""Container for structural material data."""

import dataclasses
from typing import Optional, Tuple, Union, Literal
import jax.numpy as jnp
from jax.typing import ArrayLike
import toflux.src.utils as _utils

_Ext = _utils.Extent


@dataclasses.dataclass
class ThermalMaterial:
  """Linear thermal material constants.

  Attributes:
    thermal_conductivity: The thermal conductivity of the material [W/(m*K)].
    specific_heat: Specific heat capacity of the material [J/(kg*K)].
    mass_density: Mass density of material in [kg/m^3].
    expansion_coefficient: Thermal expansion coefficient of the material [1/K].
  """

  thermal_conductivity: Optional[float] = None
  specific_heat: Optional[float] = None
  mass_density: Optional[float] = None
  expansion_coefficient: Optional[float] = None

  @property
  def diffusivity(self) -> float:
    return (self.thermal_conductivity) / (self.mass_density * self.specific_heat)


@dataclasses.dataclass
class StructuralMaterial:
  """Linear structural material constants.

  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
    yield_strength: Yield strength of the material [Pa].
  """

  youngs_modulus: Optional[float] = None
  poissons_ratio: Optional[float] = None
  mass_density: Optional[float] = None
  yield_strength: Optional[float] = None

  @property
  def shear_modulus(self) -> float:
    return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

  @property
  def bulk_modulus(self) -> float:
    return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

  @property
  def lame_parameters(self) -> tuple[float, float]:
    """Get the Lame parameters for the material.

    Returns: The Lame parameters as a tuple (lambda, mu) for the material.
    """
    lam = (
      self.youngs_modulus
      * self.poissons_ratio
      / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))
    )
    mu = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    return lam, mu


def get_lame_parameters_from_youngs_modulus_and_poissons_ratio(
  youngs_modulus: ArrayLike,
  poissons_ratio: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
  """Get the Lame parameters from Young's modulus and Poisson's ratio.

  Args:
    youngs_modulus: The Young's modulus of the material [Pa].
    poissons_ratio: The Poisson's ratio of the material [-].

  Returns: The Lame parameters as a tuple (lambda, mu) for the material.
  """
  lam = (
    youngs_modulus
    * poissons_ratio
    / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
  )
  mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
  return lam, mu


def compute_hookean_pk2(
  deformation_gradient: ArrayLike,
  lame_parameters: tuple[float, float],
):
  """Compute the PK2 stress for a Hookean material.

  This function computes the PK2 stress tensor for a Hookean material using
  the small strain approximation. The linearized material model considers the small
  strain tensor :

              epsilon = 0.5 * (grad_u + grad_u^T)
  where grad_u is the gradient of the displacement field.

  NOTE: This function uses a generic tensorial form valid for 3D, plane strain,
  or plane stress (assuming appropriate modified Lame parameters are provided).

  Args:
    deformation_gradient: The deformation gradient tensor (num_dim, num_dim).
    lame_parameters:A tuple containing the Lame parameters (lambda, mu).

  Returns: The Second Piola-Kirchhoff stress tensor (num_dim, num_dim).
  """
  lmbda, mu = lame_parameters
  eye = jnp.eye(deformation_gradient.shape[0])

  # D_ijkl = lambda * d_ij * d_kl + mu * (d_ik * d_jl + d_il * d_jk)
  elasticity_tensor = lmbda * jnp.einsum("ij, kl -> ijkl", eye, eye) + mu * (
    jnp.einsum("ik, jl -> ijkl", eye, eye) + jnp.einsum("il, jk -> ijkl", eye, eye)
  )
  grad_u = deformation_gradient - eye
  small_strain = 0.5 * (grad_u + grad_u.T)
  return jnp.einsum("ijkl, kl -> ij", elasticity_tensor, small_strain)


@dataclasses.dataclass
class FluidMaterial:
  """Linear material constants.

  Attributes:
    mass_density: Mass density of the fluid [kg/m^3].
    dynamic_viscosity: Dynamic viscosity of the fluid [Pa*s].
    thermal_conductivity: Thermal conductivity of the fluid [W/(m*K)].
    specific_heat: Specific heat capacity of the fluid [J/(kg*K)].
  """

  mass_density: Optional[float] = None
  dynamic_viscosity: Optional[float] = None
  thermal_conductivity: Optional[float] = None
  specific_heat: Optional[float] = None

  @property
  def kinematic_viscosity(self) -> float:
    return self.dynamic_viscosity / self.mass_density

  @property
  def diffusivity(self) -> float:
    """Thermal diffusivity of the fluid [m^2/s]."""
    return self.thermal_conductivity / (self.mass_density * self.specific_heat)


def brinkman_bound(dynamic_viscosity: float, out_of_plane_thickness: float) -> float:
  """Calculate the brinkman penlaty upper/lower bound for the given material constants.

  brinkman_penalty_min = 5*dynamic_viscosity/2*out_of_plane_thickness^2
  out_of_plane_thickness = 100*char_length

  brinkman_penalty_max = 5*dynamic_viscosity/2*out_of_plane_thickness^2
  out_of_plane_thickness = 0.01*char_length

  https://link.springer.com/article/10.1007/s00158-022-03420-9
  Args:
    dynamic_viscosity: The dynamic viscosity of the fluid.
    out_of_plane_thickness: The out of plane thickness of the domain.

  Returns: The brinkman penalty bound.
  """

  return 5 * dynamic_viscosity / (2 * out_of_plane_thickness**2)


def calculate_interpolation_factor(
  inv_permeability_ext: _Ext, init_inv_permeability: float, desired_mat_fraction: float
) -> float:
  """
  Calculate the interpolation factor for the penalized brinkman factor.

  interpolation_factor = (-desired_void_fraction*(inv_permeability_ext.range) +
                          inv_permeability_ext.max - init_inv_permeability)
                          /(desired_void_fraction*
                          (init_inv_permeability-inv_permeability_ext.min))

  Args:
    inv_permeability_ext: An instance of _Ext containing the extent of the
      permeability values.
    init_inv_permeability: The initial permeability value.
    desired_mat_fraction: The desired material fraction.

  Returns: The interpolation factor.
  """

  desired_void_fraction = 1.0 - desired_mat_fraction
  t_1 = desired_void_fraction * (inv_permeability_ext.range)
  t_2 = inv_permeability_ext.max - init_inv_permeability
  t_3 = desired_void_fraction * (init_inv_permeability - inv_permeability_ext.min)
  inter_factor = (-t_1 + t_2) / t_3
  return inter_factor


def compute_ramp_interpolation(
  prop: jnp.ndarray,
  ramp_penalty: float,
  prop_ext: Union["_Ext", Tuple[float, float]],
  mode: Literal["convex", "concave"] = "convex",
) -> jnp.ndarray:
  """RAMP interpolation (convex or concave).

  RAMP — Rational Approximation of Material Properties — smoothly maps the
  design field prop ∈ [0, 1] to a physical property bounded by
  prop_ext.min (fluid) and prop_ext.max (solid).

  Convex mapping (default) keeps values close to the fluid limit
    early in optimization ⟶ equation (8) in Alexandersen 2022.
    A detailed introduction to density-based topology optimisation of fluid flow problems
    with implementation in MATLAB
  Concave mapping keeps values close to the solid limit early in optimization
    ⟶ equation (7) in Marck 2013.
    Topology Optimization of Heat and Mass Transfer Problems: Laminar Flow

  The concave penalized property is calculated using the formula:

    penalised = min + (max - min)*p*(1.0 + r)/(p + r)

  and the convex penalized property is calculated using the formula:
    penalized_property = min + (max - min)*p/(1. + r_p -r_p*p)

  where:
    - p is the material property value.
    - r_p is the ramp penalty parameter.
    - min and max are the minimum and maximum values of the material property.

  Args:
  prop: Design variable array (1 = solid, 0. = fluid).
  ramp_penalty: RAMP penalty parameter r ≥ 0.
  prop_ext: Either an _Ext instance or a (min, max) tuple.
  mode: convex → fluid-biased mapping,
      concave → solid-biased mapping.

  Returns: Penalised property array with the same shape as prop.
  """

  if isinstance(prop_ext, Tuple):
    prop_ext = _Ext(min=prop_ext[0], max=prop_ext[1])

  rng = prop_ext.range

  if mode == "convex":
    denom = 1.0 + ramp_penalty - ramp_penalty * prop
    penalised = prop_ext.min + rng * prop / denom
  elif mode == "concave":
    denom = prop + ramp_penalty
    penalised = prop_ext.min + rng * prop * (1.0 + ramp_penalty) / denom
  else:
    raise ValueError(f"Unknown RAMP mode '{mode}'. Use 'convex' or 'concave'.")

  return penalised


@dataclasses.dataclass(frozen=True)
class CarreauYasudaNonNewtonianFluid:
  """Parameters for the Carreau-Yasuda non-Newtonian fluid model.

  The viscosity is modeled as dependent on the shear rate. The viscosity `eta` at
    shear rate `g` is given by:

      eta(g) = eta_inf + (eta_0 - eta_inf)*(1 + (lam*g)^a)^((n-1)/a)

  Where,
    `eta_inf` and `eta_0` are the lower and upper viscosities.
    `n` is the power law exponent
    `lam` and `a` are dimensionless parameters
  """

  eta_inf: float
  eta_0: float
  lam: float
  a: float
  n: float

  def get_viscosity(self, shear_rate: ArrayLike) -> float:
    """Returns the viscosity at the given shear rate.

    NOTE: We use the safe power function to avoid numerical issues with negative or
    zero shear rates. The safe power function returns 0 for negative or zero inputs.
    """
    del_eta = self.eta_0 - self.eta_inf
    v = 1 + _utils.safe_power(self.lam * shear_rate, self.a)
    p = (self.n - 1) / self.a
    return self.eta_inf + del_eta * (_utils.safe_power(v, p))


@dataclasses.dataclass(frozen=True)
class PowerLawNonNewtonianFluid:
  """Parameters for the power non-Newtonian fluid model.

  The viscosity is modeled as dependent on the shear rate. The viscosity `eta` at
    shear rate `g` is given by:

      eta(g) = eta_inf * (g)^n

  Where,
    `eta_inf` is the consistency Index.
    `n` is the power law exponent. (n < 1 for shear-thinning fluids) and
                                   (n > 1 for shear-thickening fluids)
  """

  eta_inf: float
  n: float

  def get_viscosity(self, shear_rate: ArrayLike) -> float:
    """Returns the viscosity at the given shear rate.

    NOTE: We use the safe power function to avoid numerical issues with negative or
    zero shear rates. The safe power function returns 0 for negative or zero inputs.
    """
    return self.eta_inf * _utils.safe_power(shear_rate, self.n)

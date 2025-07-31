"""Utility functions."""

from typing import Literal, Union, Sequence, Callable, Tuple
import enum
import chex
import numpy as np
import scipy.special as spy_spl
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
from jax.typing import ArrayLike


class Direction(enum.Enum):
  """Euclidean Directions."""

  X = 0
  Y = 1
  Z = 2


@chex.dataclass
class Extent:
  """Extent of a variable."""

  min: ArrayLike
  max: ArrayLike

  @property
  def range(self) -> ArrayLike:
    return self.max - self.min

  @property
  def center(self) -> ArrayLike:
    return 0.5 * (self.min + self.max)

  def normalize_array(self, x: ArrayLike) -> ArrayLike:
    """Linearly normalize `x` using `extent` ranges."""
    return (x - self.min) / self.range

  def renormalize_array(self, x: ArrayLike) -> ArrayLike:
    """Recover array from linearly normalized `x` using `extent` ranges."""
    return x * self.range + self.min


def safe_power(x: jax.Array, exp: float) -> jax.Array:
  """Compute the power `x**exp` with a safe check for negative/zero values.

  This function ensures that the input `x` is positive before applying the power
    operation. If `x` is negative or zero, it returns zero. This ensures that the
  power operation does not result in undefined behavior or complex numbers.

  Args:
    x: Input array.
    exp: Exponent value.

  Returns: The result of `x**exp` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.power(z, exp), 0.0)


def safe_log(x: jax.Array) -> jax.Array:
  """Compute the natural logarithm of `x` with a safe check for non-positive values.

  Args:
    x: Input array.
  Returns: The natural logarithm of `x` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.log(z), 0.0)


def safe_sqrt(x: jax.Array) -> jax.Array:
  """Compute the square root of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    square root operation. If `x` is negative, it returns zero. This ensures that
    the square root operation does not result in undefined behavior.

  Args:
    x: Input array.

  Returns: The square root of `x` if `x` is non-negative, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x >= 0.0, jnp.sqrt(z), 0.0)


def safe_divide(x: jax.Array, y: jax.Array, eps: float = 1.0e-6) -> jax.Array:
  """Compute the division of x by y with a safe check for division by zero.

  This function ensures that the denominator `y` is non-zero before applying the
    division operation. If `y` is zero, it returns zero. This ensures that the
    division operation does not result in undefined behavior.

  Args:
    x: Numerator array.
    y: Denominator array.
    eps: Small value below which the absolute value of the denominator is
      treated as zero.

  Returns: The result of `x / y` if `y` is non-zero, otherwise zero.
  """
  z = jnp.where(jnp.abs(y) < eps, 1.0, y)
  return jnp.where(jnp.abs(y) < eps, 0.0, x / z)


def safe_pnorm(x: jax.Array, p: float, axis: int):
  """Compute the p-norm of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    p-norm operation. If `x` is negative, it returns zero. This ensures that the
  p-norm operation does not result in undefined behavior.

  The p-norm is defined as:
               ||x||_p = (sum_i(|x_i|^p))^(1/p)

  The function is often used to compute a smooth approximation to the maximum (or minimum)
  of a set of values.

  Args:
    x: The input array.
    p: The Exponent value. The larger the value, the closer the p-norm is to the maximum.
      However, the problem becomes more nonlinear. A typical value is 6.0.
    axis: The axis along which the p-norm is computed.

  Returns: The p-norm of `x` computed in a safe manner along the specified axis.
  """
  sum_x = jnp.sum(safe_power(x, p), axis=axis)
  return safe_power(sum_x, 1.0 / p)


def inverse_sigmoid(y: jax.Array) -> jax.Array:
  """The inverse of the sigmoid function.

  The sigmoid function f:x->y is defined as:

           f(x) = 1 / (1 + exp(-x))

  The inverse sigmoid function g: y->x is defined as:

           g(y) = ln(y / (1 - y))

  For details see https://tinyurl.com/y7mr76hm
  """
  return jnp.log(y / (1.0 - y))


def smooth_extremum(
  x: jax.Array,
  order: float = 100.0,
  extreme: Literal["min", "max"] = "min",
  axis: Union[int, Sequence[int], None] = None,
) -> jax.Array:
  """Compute the smooth (approximate) minimum/maximum of an array.

  The function approximates the minimum/maximum of an array using the logsumexp
  function. The function is often used to compute a smooth approximation to the
  maximum (or minimum) of a set of values maintaining differentiability.

  Args:
    x: Array of whose entries we wish to compute the minimum.
    order: A float that ensures that the values are scaled appropriately to
      ensure no numerical overflow/underflow. Further, depending upon the
      magnitudes of the entry, experimenting with different values of `order`
      can result in better answers.
    extreme: Whether we wish to compute the minima or the maxima.
    axis: The axis along which the extremum is computed. If None, the extremum
      is computed over the entire array.
  """
  scale = jnp.amax(jnp.abs(jax.lax.stop_gradient(x))) / order
  sgn = -1.0 if extreme == "min" else 1.0
  return scale * sgn * jax.scipy.special.logsumexp(sgn * x / scale, axis=axis)


def gauss_integ_points_weights(
  order: int,
  dimension: int,
) -> tuple[jax.Array, jax.Array]:
  """
  Returns the Gauss integration points and weights for the given order and
    dimension. The number of gauss points is order^dimension.

  Args:
    order (int): The order of the Gauss quadrature.
    dimension (int): The dimension of the integration. Must be in range (1, 3).

  Returns: A tuple containing the integration points and weights.
    - points (numpy.ndarray): An array of shape (order^dimension, dimension)
       containing the integration points.
    - weights (numpy.ndarray): An array of shape (order^dimension,) containing the
        integration weights.

  Raises:
      ValueError: If dimension is not in (1, 3).
  """
  # Get 1D Gauss points and weights
  x, w = spy_spl.roots_legendre(order)

  if dimension == 1:
    points = x.reshape(-1, 1)
    weights = w

  elif dimension == 2:
    # Generate 2D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j] for x_i in x for x_j in x])
    weights = jnp.array([w_i * w_j for w_i in w for w_j in w])

  elif dimension == 3:
    # Generate 3D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j, x_k] for x_i in x for x_j in x for x_k in x])
    weights = jnp.array([w_i * w_j * w_k for w_i in w for w_j in w for w_k in w])

  else:
    raise ValueError("Dimension must be in (1, 3)")

  return points, weights


def threshold_filter(density: jax.Array, beta: float, eta: float = 0.5) -> jax.Array:
  """Threshold project the density, pushing the values towards 0/1.

  Args:
    density: Array of size (num_elems,) that are in [0,1] that contain the
      density of the elements.
    beta: Sharpness of projection (typically ~ 1-32). Larger value indicates
      a sharper projection.
    eta: Center value about which the values are projected.

  Returns: The thresholded density value array of size (num_elems,).
  """
  v1 = jnp.tanh(eta * beta)
  nm = v1 + jnp.tanh(beta * (density - eta))
  dnm = v1 + jnp.tanh(beta * (1.0 - eta))
  return nm / dnm


def _cdist(a: jax.Array, b: jax.Array) -> jax.Array:
  """Computes pairwise Euclidean distances between rows of a and b.

  Args:
    a: Array of shape (N, D) - N points, D dimensions.
    b: Array of shape (M, D) - M points, D dimensions.

  Returns:
    Array of shape (N, M) containing distances.
  """
  diff = a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :]
  return jnp.linalg.norm(diff, axis=-1)


class Filters(enum.Enum):
  """Enum for different types of filters."""

  LINEAR = enum.auto()
  CIRCULAR = enum.auto()
  GAUSSIAN = enum.auto()


def create_density_filter(
  coords: jax.Array,
  cutoff_distance: float,
  filter_type: Filters = Filters.LINEAR,
  eps: float = 1e-12,
) -> jax_sprs.BCOO:
  """Creates a density filter to smoothen out the field.

  The density filter is to ensure that the obtained density fields do not have
  checkerboard patterns. This is common in density-based topology optimization problems.

  Args:
    coords: An array of shape (num_pts, num_dim) of the coordinates of the points.
    cutoff_distance: A float, the radius beyond which the filter has zero influence.
    filter_type: A string, one of 'linear', 'circular', or 'gaussian'.
    eps: A float, small value to avoid division by zero added to the entries.

  Returns: A BCOO sparse matrix of size (num_pts, num_pts) of the filter.
  """
  num_pts = coords.shape[0]

  distances = _cdist(coords, coords)

  row_indices, col_indices = jnp.where(distances <= cutoff_distance)
  relevant_distances = distances[row_indices, col_indices]

  if filter_type == Filters.LINEAR:
    filter_values = 1.0 - (relevant_distances / cutoff_distance)

  elif filter_type == Filters.CIRCULAR:
    filter_values = jnp.sqrt(1.0 - (relevant_distances / cutoff_distance) ** 2)

  elif filter_type == Filters.GAUSSIAN:
    sigma = cutoff_distance / 3.0
    filter_values = jnp.exp(-0.5 * (relevant_distances / sigma) ** 2)

  else:
    raise ValueError(f"Unsupported filter type: {filter_type.name}")

  row_sums = jnp.zeros(num_pts).at[row_indices].add(filter_values) + eps
  inv_row_sums = 1.0 / row_sums
  normalized_filter_values = filter_values * inv_row_sums[row_indices]

  return jax_sprs.BCOO(
    (normalized_filter_values, jnp.stack([row_indices, col_indices], axis=1)),
    shape=(num_pts, num_pts),
  )


class RBFInterpolator:
  def __init__(self, x: jax.Array, values: jax.Array, kernel: Callable) -> None:
    """Radial Basis Function (RBF) interpolator.

    Args:
      x: Array of (num_pts, num_dim) - Coordinates of known values.
      values: Array of (num_pts,) - Values at known coordinates.
      kernel: A callable kernel function (epsilon pre-set via partial).
    """
    self.x = x
    self.values = values
    self.kernel = kernel
    self.coefficients = self._compute_coefficients()

  def _compute_coefficients(self) -> jax.Array:
    """Calculate the RBF coefficients."""
    pairwise_distances = _cdist(self.x, self.x)
    gram_matrix = self.kernel(pairwise_distances)
    jitter = 1e-8
    eye_matrix = jnp.eye(gram_matrix.shape[0])
    coefficients = jnp.linalg.solve(gram_matrix + jitter * eye_matrix, self.values)
    return coefficients

  def interpolate(self, x_new: jax.Array) -> jax.Array:
    """Interpolate values at new locations using the RBF."""
    distances_to_known = _cdist(x_new, self.x)
    weights = self.kernel(distances_to_known)
    interpolated_values = jnp.dot(weights, self.coefficients)
    return interpolated_values


def box_car(
  extent: Tuple[jax.Array, jax.Array],
  x: jax.Array,
  sharpness: float = 10.0,
  exterior_val: float = 0.0,
  interior_val: float = 1.0,
) -> jax.Array:
  """Compute n-dimensional box car.

          Box_car(x) = {  interior_val, start ~< x ~< end
                       {  exterior_val, otherwise
  Args:
    extent: A tuple containing the start and end points of the boxcar. Each
      entry is an array of (num_dim,).
    x: Array of (num_pts, num_dim) of the points onto which we would like to
      compute the boxcar function.
    sharpness: To make the function differentiable, we utilize a smooth
      transition function between 1 and 0 rather than a hard cut-off. This
      value controls the sharpness of the transition. Higher values result in
      a sharper transition. We normalize the entries by the provided extent so
      that the sharpness value is independent of the scale of `x`.
    exterior_val: The value the function assumes outside the boxcar interval.
    interior_val: The value the function assumes inside the boxcar interval.

  Returns: Array of (num_pts,) with values in  [exterior_val, interior_val] of
    the boxcar values at given points.
  """
  assert jnp.all(
    extent[0] < extent[1]
  ), "All start points should be less than end points."
  sharpness_norm = sharpness / jnp.abs(0.5 * (extent[0] + extent[1]))

  start_norm = jnp.einsum("d, pd -> pd", sharpness_norm, x - extent[0][None, :])
  y_start = jax.nn.sigmoid(start_norm)

  end_norm = jnp.einsum("d, pd -> pd", sharpness_norm, x - extent[1][None, :])
  y_end = 1.0 - jax.nn.sigmoid(end_norm)

  unit_boxcar = jnp.prod(y_start, axis=1) * jnp.prod(y_end, axis=1)
  return exterior_val + (interior_val - exterior_val) * unit_boxcar


def is_point_on_segment(
  start_pt: np.ndarray, end_pt: np.ndarray, pt: np.ndarray, tolerance: float = 1e-9
) -> bool:
  """Checks if a point lies on a line segment with a given tolerance.

  A point is on the segment if it is both collinear with the segment's
  endpoints and lies within the axis-aligned bounding box of the segment.

  Args:
    start_pt: Array of shape (n,) of the start point of the line segment.
    end_pt: Array of shape (n,) of the end point of the line segment.
    pt: Array of shape (n,) of the point to check.
    tolerance: A small value to account for floating-point inaccuracies.

  Returns: True if the point is on the line segment, False otherwise.
  """
  # Boundedness Check
  in_bounds = np.all(pt >= np.minimum(start_pt, end_pt) - tolerance) and np.all(
    pt <= np.maximum(start_pt, end_pt) + tolerance
  )
  if not in_bounds:
    return False

  # Collinearity Check
  cross_product = np.cross(end_pt - start_pt, pt - start_pt)
  return np.abs(cross_product) < tolerance

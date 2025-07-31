"""Topology Optimization using Neural Networks (TOuNN) network module.

For more details, see:

Chandrasekhar, Aaditya, and Krishnan Suresh. "TOuNN: Topology optimization using neural
networks." Structural and Multidisciplinary Optimization 63, no. 3 (2021): 1135-1149.
"""

from typing import Callable, Sequence, Optional
import dataclasses
import numpy as np
import jax.numpy as jnp
from flax import nnx


class TopNet(nnx.Module):
  def __init__(
    self, num_neurons: Sequence[int], rngs: nnx.Rngs, use_batch_norm: bool = True
  ):
    """Defines a simple fully connected neural network.

    Args:
      num_neurons: The number of neurons in each layer.
      rngs: The random number generator.
      use_batch_norm: Whether to use batch normalization.
    """
    self.use_batch_norm = use_batch_norm
    self.layers = []
    if use_batch_norm:
      self.bn_layers = []
    for i in range(len(num_neurons) - 1):
      linear = nnx.Linear(num_neurons[i], num_neurons[i + 1], rngs=rngs)
      self.layers.append(linear)
      if use_batch_norm:
        self.bn_layers.append(nnx.BatchNorm(num_neurons[i + 1], rngs=rngs))

  def __call__(self, x: jnp.ndarray, activation: Callable = nnx.relu) -> jnp.ndarray:
    """Defines the forward pass of the network.

    In this context of TOuNN, x corresponds to the coordinates of a mesh, (or in
    the case of Fourier activates, coordinates of the mesh projected to frequency
    space.)

    Args:
      x: The input to the network of size (num_pts, num_input_dim).
      activation: The activation function to use.

    Returns:
      The output of the network of size(num_pts, num_output_dim) .
    """
    for i in range(len(self.layers) - 1):
      x = self.layers[i](x)
      if self.use_batch_norm:
        x = self.bn_layers[i](x)
      x = activation(x)
    return self.layers[-1](x)


@dataclasses.dataclass
class FourierProjection:
  """Parameters for the Fourier projection.

  In the Fourier projection, the input coordinates are projected to frequency space
  using a Fourier map. The Fourier map is a matrix of size (num_dim, num_terms) where
  num_dim is the number of dimensions and num_terms is the number of Fourier terms.

  For more details, see:

  Chandrasekhar, Aaditya, and Krishnan Suresh. "Approximate length scale filter in 
  topology optimization using fourier enhanced neural networks." CAD 150 (2022): 103277.

  Attributes:
    num_input_dim: The number of physical euclidean input space dimensions.
    num_terms: The number of Fourier terms.
    max_radius: The maximum radius of the Fourier terms.
    min_radius: The minimum radius of the Fourier terms.
    random_seed: The random seed for reproducibility.

  Upon initialization, the Fourier map of size (num_input_dim, num_terms) is generated.
  """

  num_input_dim: int
  num_terms: int
  max_radius: float
  min_radius: float
  random_seed: Optional[int] = 0

  def __post_init__(self):
    rng = np.random.default_rng(self.random_seed)
    freq_sign = rng.choice([-1.0, 1.0], (self.num_input_dim, self.num_terms))
    w_min, w_max = 1.0 / (2 * self.max_radius), 1.0 / (2 * self.min_radius)
    freq = rng.uniform(w_min, w_max, (self.num_input_dim, self.num_terms))
    self.coordn_map = jnp.einsum("ij,ij->ij", freq_sign, freq)

  def apply(self, euclidean_coords: jnp.ndarray) -> jnp.ndarray:
    """Applies the Fourier projection to the input Euclidean coordinates.

    Args:
      euclidean_coords: The input coordinates of size (num_pts, num_input_dim).

    Returns: The projected coordinates of size (num_pts, 2 * num_terms). The first
      num_terms columns correspond to the cosine terms and the next num_terms columns
      correspond to the sine terms.
    """
    # (p)oints, (e)uclidean, (f)ourier
    c = jnp.cos(2 * np.pi * jnp.einsum("pe,ef->pf", euclidean_coords, self.coordn_map))
    s = jnp.sin(2 * np.pi * jnp.einsum("pe,ef->pf", euclidean_coords, self.coordn_map))
    return jnp.concatenate((c, s), axis=1)


@dataclasses.dataclass
class Symmetry:
  """Parameters for the symmetry projection.

  In the symmetry projection, the input coordinates are projected to a space with
  respect to the symmetry of the problem.

  Attributes:
    sym_yz_coord: The x-coordinate of the symmetry plane along the YZ plane.
    sym_xz_coord: The y-coordinate of the symmetry plane along the XZ plane.
    sym_xy_coord: The z-coordinate of the symmetry plane along the XY plane.
  """

  sym_yz_coord: Optional[float] = None
  sym_xz_coord: Optional[float] = None
  sym_xy_coord: Optional[float] = None

  def apply(self, coords: jnp.ndarray) -> jnp.ndarray:
    """Applies the symmetry projection to the input Euclidean coordinates.

    Args:
      coords: The input Euclidean coordinates of size (num_pts, num_input_dim).

    Returns: The projected coordinates of size (num_pts, num_input_dim) that are
      symmetrized.
    """
    if self.sym_yz_coord is not None:
      coords = coords.at[:, 0].set(
        self.sym_yz_coord + jnp.abs(coords[:, 0] - self.sym_yz_coord)
      )
    if self.sym_xz_coord is not None:
      coords = coords.at[:, 1].set(
        self.sym_xz_coord + jnp.abs(coords[:, 1] - self.sym_xz_coord)
      )
    if self.sym_xy_coord is not None:
      coords = coords.at[:, 2].set(
        self.sym_xy_coord + jnp.abs(coords[:, 2] - self.sym_xy_coord)
      )
    return coords

"""Linear and Nonlinear solver for the system of equations integrated into JAX."""

import warnings
from typing import Dict
import abc
import enum
import functools
import numpy as np
import scipy.sparse as spy_sprs
import scipy.sparse.linalg as spy_linalg
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
from jax.typing import ArrayLike
import pyamg
import petsc4py.PETSc as PETSc

try:
  import pypardiso  # type: ignore
except ImportError:
  warnings.warn("pypardiso library not found. Some solvers may not be available.")


class LinearSolvers(enum.Enum):
  """Linear solvers wrapped around with `jax.lax.custom_linear_solve`."""

  LINALG_SOLVE = enum.auto()
  AMG_CG = enum.auto()
  AMG_BICGSTAB = enum.auto()
  SCIPY_SPARSE = enum.auto()
  PARDISO = enum.auto()
  PETSC = enum.auto()


def _jacobi_preconditioner(A: spy_sprs.coo_matrix) -> spy_sprs.coo_matrix:
  """Computes the Jacobi preconditioner for a sparse matrix A in COO format.

  Args:
    A: The sparse matrix in COO format.

  Returns: The Jacobi preconditioner in COO format.
  """

  # Extract diagonal elements
  diag_data = A.diagonal()

  # Ensure no zeros on the diagonal (avoid division by zero)
  diag_data[diag_data == 0.0] = 1.0

  diag_idxs = np.arange(A.shape[0])
  # Construct the preconditioner as a diagonal matrix
  M_inv = spy_sprs.coo_matrix((1.0 / diag_data, (diag_idxs, diag_idxs)), shape=A.shape)
  return M_inv


def _petsc_solve(
  A: spy_sprs.csr_matrix, b: ArrayLike, solver_options: Dict
) -> np.ndarray:
  """Solve for u = A^{-1}b using PETSc.

  Args:
    A: A sparse CSR matrix of shape (m,m).
    b: The rhs vector of shape (m,).
    solver_options: A dictionary containing PETSc solver options.

  Returns: The solution vector of shape (m,).
  """
  ksp_type = (
    solver_options["petsc_solver"]["ksp_type"]
    if "ksp_type" in solver_options["petsc_solver"]
    else "bcgsl"
  )
  pc_type = (
    solver_options["petsc_solver"]["pc_type"]
    if "pc_type" in solver_options["petsc_solver"]
    else "ilu"
  )

  A = PETSc.Mat().createAIJ(
    size=A.shape,
    csr=(
      A.indptr.astype(PETSc.IntType, copy=False),
      A.indices.astype(PETSc.IntType, copy=False),
      A.data,
    ),
  )

  rhs = PETSc.Vec().createSeq(len(b))
  rhs.setValues(range(len(b)), np.array(b))
  ksp = PETSc.KSP().create()
  ksp.setOperators(A)
  ksp.setFromOptions()
  ksp.setType(ksp_type)
  ksp.pc.setType(pc_type)

  if ksp_type == "tfqmr":
    ksp.pc.setFactorSolverType("mumps")

  x = PETSc.Vec().createSeq(len(b))
  ksp.solve(rhs, x)

  return x.getArray()


def solve(
  A: jax_sprs.BCOO,
  b: jnp.ndarray,
  params: Dict,
  u0: jnp.ndarray = None,
) -> jnp.ndarray:
  """Solve for u = A^{-1}b using custom solvers.

  This function uses `jax.lax.custom_linear_solve` to solve the linear system of
    equations defined by the sparse matrix A and the right-hand side vector b. For
    the list of available solvers, see `LinearSolvers` enum. The solvers themselves
    are not differentiable. By wrapping them in `jax.lax.custom_linear_solve`, we can
    use them in JAX computations while still allowing for differentiation through the
    linear solve operation.

  Args:
    A: A sparse BCOO matrix of shape (m,m)
    b: The rhs vector of shape (m,).
    params: Additional parameters for the solver. This should include:
      - "solver": The type of solver to use, as defined in `LinearSolvers`
      - other solver-specific parameters, e.g., "rtol" for relative tolerance.
    u0: The initial guess for the solution of shape (m,). Not required for all solvers,
      but some solvers may need it (e.g., iterative solvers like CG or BICGSTAB).

  Returns: The solution vector of shape (m,).
  """

  def mv(u):
    return A @ u

  def solver_wrapper(A, b):
    A_sp = spy_sprs.coo_matrix(
      (jax.lax.stop_gradient(A.data), (A.indices[:, 0], A.indices[:, 1])), shape=A.shape
    )
    b = jax.lax.stop_gradient(b)

    if params["solver"] == LinearSolvers.AMG_CG:
      M = _jacobi_preconditioner(A_sp)
      x, _ = pyamg.krylov.cg(A_sp, b, tol=params["rtol"], x0=u0, M=M)

    elif params["solver"] == LinearSolvers.AMG_BICGSTAB:
      M = _jacobi_preconditioner(A_sp)
      x, _ = pyamg.krylov.bicgstab(A_sp, b, tol=params["rtol"], x0=u0, M=M)

    elif params["solver"] == LinearSolvers.SCIPY_SPARSE:
      x = spy_linalg.spsolve(A_sp.tocsr(), b)

    elif params["solver"] == LinearSolvers.LINALG_SOLVE:
      x = jnp.linalg.solve(A, b)

    elif params["solver"] == LinearSolvers.PARDISO:
      x = pypardiso.spsolve(A_sp.tocsr(), np.asarray(b))

    elif params["solver"] == LinearSolvers.PETSC:
      x = _petsc_solve(A_sp.tocsr(), np.asarray(b), params)

    else:
      raise ValueError("Invalid solver type")

    return x.astype(b.dtype).reshape(b.shape)

  result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

  def cust_fwd_solver(mv, b):
    return jax.pure_callback(solver_wrapper, result_shape, A, b)

  def cust_bwd_solver(mv, b):
    return jax.pure_callback(solver_wrapper, result_shape, A.T, b)

  sol = jax.lax.custom_linear_solve(
    mv, b, solve=cust_fwd_solver, transpose_solve=cust_bwd_solver
  )
  return sol.reshape(-1)


class NonlinearProblem(abc.ABC):
  """Base class for the nonlinear problems.

  This class defines the interface for nonlinear problems that can be solved using
    Newton's method or modified Newton's method. The derived classes should implement
    the `get_residual_and_tangent_stiffness` method to compute the residual and
    tangent stiffness matrix of the system of equations.
  """

  def __init__(self, solver_settings: dict):
    """Initializes the nonlinear problem with solver settings.

    Args:
      solver_settings: Dictionary containing the solver settings.
    """
    self.solver_settings = solver_settings

  @abc.abstractmethod
  def get_residual_and_tangent_stiffness(
    self, x: jax.Array, *params
  ) -> tuple[jax.Array, jax_sprs.BCOO]:
    """Base class function for computing the residual and tangent stiffness matrix.
    The function takes as arguments (x0, *params) where x0 is the current guess
    of the solution and *params are the additional parameters.

    Returns:
      res: An array of (num_dofs,) containing the residual.
      K: A sparse matrix of size (num_dofs, num_dofs) containing the tangent
        stiffness matrix.
    """
    pass


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def newton_raphson_solve(
  problem: NonlinearProblem,
  x0: jnp.ndarray,
  *params,
) -> jnp.ndarray:
  """Nonlinear solver using Newton's method.

  This function implements the Newton-Raphson method to solve the nonlinear system of
    equations defined by the `problem` object. The method iteratively updates the
    solution until convergence is achieved based on the specified settings in
    `problem.solver_settings["nonlinear"]`.

    We then define a custom JVP rule for this function to enable differentiation
    through the nonlinear solve operation. In particular, we use the implicit function
    theorem to compute the JVP of the solution with respect to the parameters.
    This avoids the need to differentiate through the entire Newton-Raphson iteration
    chain, which can be computationally expensive and memory-intensive.

  Args:
    problem: The function to compute the residual.
    jacobian_fn: The function to compute the jacobian. The function takes the
      same arguments as residual_fn.
    x0: Array of (n,) of the initial guess.
    *params: Additional parameters for the residual and jacobian functions.

  Returns: Array of (n,) of the solution.
  """
  ctr = 0
  res, _ = problem.get_residual_and_tangent_stiffness(x0, *params)
  init_res_norm = jax.lax.stop_gradient(jnp.linalg.norm(res))
  state = (x0, init_res_norm, ctr)

  settings = problem.solver_settings["nonlinear"]

  def cond_fn(state):
    _, res_norm, ctr = state
    cond_iter = ctr < settings["max_iter"]
    cond_res = res_norm > settings["threshold"] * init_res_norm
    return cond_iter & cond_res

  def body_fn(state):
    x0, res_norm, ctr = state
    residual, jacobian = problem.get_residual_and_tangent_stiffness(x0, *params)

    x0 -= solve(jacobian, residual, params=problem.solver_settings["linear"])
    res_norm = jax.lax.stop_gradient(jnp.linalg.norm(residual))
    ctr += 1

    return (x0, res_norm, ctr)

  x0, _, ctr = jax.lax.while_loop(cond_fn, body_fn, state)
  jax.debug.print("NR converged in {x} iters", x=ctr)
  return x0


@newton_raphson_solve.defjvp
def solve_jvp(
  problem: NonlinearProblem,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """JVP rule for the Newton-Raphson solver.

  This function implements the JVP rule for the Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters.

  Args:
    problem: The nonlinear problem object.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the solution x and the JVP of the solution with respect
    to the parameters.
  """
  x0, *params = primals
  _, *dparams = tangents

  x = newton_raphson_solve(problem, x0, *params)
  _, dr_dp, jacobian_dr_dx = jax.jvp(
    problem.get_residual_and_tangent_stiffness,
    (x, *params),
    (jnp.zeros_like(x), *dparams),
    has_aux=True,
  )
  jinv_v_dr_dp = solve(jacobian_dr_dx, -dr_dp, params=problem.solver_settings["linear"])
  return x, jinv_v_dr_dp


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def modified_newton_raphson_solve(
  problem: NonlinearProblem,
  x0: jnp.ndarray,
  *params,
) -> jnp.ndarray:
  """Modified Newton-Raphson solver for nonlinear problems.

  This function implements a modified version of the Newton-Raphson method that
    uses a half-step approach to improve convergence. The method iteratively updates the
    solution until convergence is achieved based on the specified settings in
    `problem.solver_settings["nonlinear"]`.

  Args:
    problem: The nonlinear problem object that defines the system of equations.
    x0: Initial guess for the solution, an array of shape (num_dofs,).
    *params: Additional parameters required by the problem's residual and tangent
      stiffness functions.

  Returns: The solution vector of shape (num_dofs,).
  """

  ctr = 0
  init_res, _ = problem.get_residual_and_tangent_stiffness(x0, *params)
  init_res_norm = jax.lax.stop_gradient(jnp.linalg.norm(init_res))
  state = (x0, init_res_norm, ctr)
  settings = problem.solver_settings["nonlinear"]

  def cond_fn(state):
    _, res_norm, ctr = state
    cond_iter = ctr < settings["max_iter"]
    cond_res = res_norm > settings["threshold"] * init_res_norm
    return cond_iter & cond_res

  def body_fn(state):
    x0, _, ctr = state
    # current step
    residual, jacobian = problem.get_residual_and_tangent_stiffness(x0, *params)

    res_norm_curr = jax.lax.stop_gradient(jnp.linalg.norm(residual))
    delta_x = solve(jacobian, residual, params=problem.solver_settings["linear"])

    # half step
    x0_half_step = x0 - 0.5 * delta_x
    residual_half_step, _ = problem.get_residual_and_tangent_stiffness(
      x0_half_step, *params
    )
    res_norm_half_step = jax.lax.stop_gradient(jnp.linalg.norm(residual_half_step))

    # full step
    x0_full_step = x0 - delta_x
    residual_full_step, _ = problem.get_residual_and_tangent_stiffness(
      x0_full_step, *params
    )
    res_norm_full_step = jax.lax.stop_gradient(jnp.linalg.norm(residual_full_step))

    lam = (3 * res_norm_curr + res_norm_full_step - 4 * res_norm_half_step) / (
      4 * res_norm_curr + 4 * res_norm_full_step - 8 * res_norm_half_step
    )
    lam = jnp.clip(lam, max=1.0, min=0.01)
    x0 = x0 - lam * delta_x

    ctr += 1
    return (x0, res_norm_curr, ctr)

  x0, res_norm, ctr = jax.lax.while_loop(cond_fn, body_fn, state)
  jax.debug.print(
    "NR converged in {x} iters, res_norm/res_norm_0: {res_norm}",
    x=ctr,
    res_norm=jax.lax.stop_gradient(jnp.linalg.norm(res_norm) / init_res_norm),
  )

  return x0


@modified_newton_raphson_solve.defjvp
def solve_jvp(  # noqa: F811
  problem: NonlinearProblem,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """JVP rule for the modified Newton-Raphson solver.

  This function implements the JVP rule for the modified Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters.

  Args:
    problem: The nonlinear problem object.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the solution x and the JVP of the solution with respect
    to the parameters.
  """
  x0, *params = primals
  _, *dparams = tangents

  x = modified_newton_raphson_solve(problem, x0, *params)
  _, df_dp, jacobian = jax.jvp(
    problem.get_residual_and_tangent_stiffness,
    (x, *params),
    (jnp.zeros_like(x), *dparams),
    has_aux=True,
  )
  jinvv_dfdp = solve(jacobian, -df_dp, params=problem.solver_settings["linear"])

  return x, jinvv_dfdp

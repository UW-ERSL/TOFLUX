"""Plotting utilities for the mesh and fields."""

from typing import Optional, List
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap

import toflux.src.geometry as _geom
import toflux.src.mesher as _mesh
import toflux.src.bc as _bc


def plot_grid_mesh(
  mesh: _mesh.Mesh,
  field: Optional[np.ndarray] = None,
  ax: plt.Axes = None,
  cmap: str = "coolwarm",
  edge_color: str = "none",
  val_range: tuple[float, float] = None,
  label: str = None,
  colorbar: bool = True,
) -> plt.Axes:
  """Plot the field on the mesh.

  Args:
    mesh: The mesh object.
    field: The field to plot. If None, only the mesh is plotted.
    ax: The matplotlib axes to plot on. If None, a new figure is created.
    cmap: The colormap to use.
    edge_color: The color of the edges of the elements.
    val_range: Optional tuple of (min_val, max_val) to set the color range. If None,
      the range is determined from the field.
    label: The label for the colorbar. If None, no label is added.
    colorbar: Whether to add a colorbar. If False, no colorbar is added.
  Returns:
    ax: The matplotlib axes with the plotted mesh and field.
  """
  if ax is None:
    _, ax = plt.subplots()

  element_coords = mesh.nodes.coords[mesh.elem_nodes]
  polygons = [plt.Polygon(coords, closed=True) for coords in element_coords]
  p = PatchCollection(polygons, edgecolor=edge_color, linewidth=1)

  if field is not None:
    assert field.shape[0] == mesh.num_elems, (
      "The field must have the same number of elements as the mesh."
    )

    if val_range is not None:
      field_min, field_max = val_range
    else:
      field_min = np.min(field) if field.size > 0 else 0
      field_max = np.max(field) if field.size > 0 else 1
      if field_min == field_max:
        field_min -= 0.5
        field_max += 0.5

    norm = plt.Normalize(vmin=field_min, vmax=field_max)
    p.set_array(field)
    p.set_norm(norm)
    p.set_cmap(cmap)
    ax.add_collection(p)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(field)
    if colorbar:
      plt.colorbar(sm, ax=ax, label=label)

  else:
    p.set_facecolor("none")
    ax.add_collection(p)

  ax.autoscale_view()
  ax.set_aspect("equal", adjustable="box")
  return ax


def plot_bc(
  bcs: List[_bc.bcLike],
  mesh: _mesh.Mesh,
  ax: plt.Axes = None,
  linewidth: float = 2.0,
):
  """Plot boundary conditions on the mesh.

  Args:
      bcs: List of BoundaryCondition objects.
      mesh: Mesh object.
      ax: Matplotlib Axes. If None, a new figure is created.
      linewidth: Width of the boundary lines.
  """
  if ax is None:
    _, ax = plt.subplots()

  color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
  bc_colors = {}
  for bc in bcs:
    if bc.name not in bc_colors:
      bc_colors[bc.name] = next(color_cycle)

  # Group faces by (bc_name, bc_type)
  bc_face_groups = {}  # (bc_name, bc_type) -> list of (elem, face)
  for bc in bcs:
    key = (bc.name, bc.type.name)
    if key not in bc_face_groups:
      bc_face_groups[key] = []
    bc_face_groups[key].extend([tuple(ef) for ef in bc.elem_faces])

  bc_faces_set = set()
  for faces in bc_face_groups.values():
    bc_faces_set.update(faces)

  # All boundary faces
  face_nodes = mesh.elem_template.face_connectivity
  boundary_faces = np.argwhere(mesh.boundary_faces)
  boundary_faces_set = set((int(e), int(f)) for e, f in boundary_faces)

  # Plot each BC group in batch
  for (bc_name, bc_type), faces in bc_face_groups.items():
    if not faces:
      continue
    color = bc_colors[bc_name]
    linestyle = "-" if bc_type == "DIRICHLET" else "--"
    # Collect all x and y for all faces
    x_lines = []
    y_lines = []
    for elem, face in faces:
      nodes = mesh.elem_nodes[elem, face_nodes[face]]
      coords = mesh.nodes.coords[nodes]
      x_lines.append(coords[:, 0])
      y_lines.append(coords[:, 1])
    # Plot all at once
    ax.plot(
      np.array(x_lines).T,
      np.array(y_lines).T,
      color=color,
      linestyle=linestyle,
      linewidth=linewidth,
      label=bc_name,
    )

  # Plot remaining boundary faces (not in any BC) as black solid
  other_faces = list(boundary_faces_set - bc_faces_set)
  if other_faces:
    x_lines = []
    y_lines = []
    for elem, face in other_faces:
      nodes = mesh.elem_nodes[elem, face_nodes[face]]
      coords = mesh.nodes.coords[nodes]
      x_lines.append(coords[:, 0])
      y_lines.append(coords[:, 1])
    ax.plot(
      np.array(x_lines).T,
      np.array(y_lines).T,
      color="black",
      linestyle="-",
      linewidth=linewidth,
      label="_other_",
    )

  # Remove duplicate legend entries
  handles, labels = ax.get_legend_handles_labels()
  unique = dict()
  for h, l in zip(handles, labels):
    if l != "_other_" and l not in unique:
      unique[l] = h
  ax.legend(unique.values(), unique.keys(), loc="best", fontsize=12)

  ax.set_aspect("equal", adjustable="box")
  ax.set_title("Boundary Conditions")
  ax.axis("off")
  return ax


def plot_brep(
  brep: _geom.BrepGeometry,
  ax: plt.Axes = None,
):
  """Plot the edges and nodes extracted from a BREP geometry."""
  if ax is None:
    _, ax = plt.subplots()

  # Plot edges
  for edge_no, edge_nodes in brep.edges.items():
    x_coords = [edge_nodes[0][0], edge_nodes[1][0]]
    y_coords = [edge_nodes[0][1], edge_nodes[1][1]]

    ax.plot(
      x_coords,
      y_coords,
      marker="o",
      linestyle=":",
      color="blue",
    )

    mid_x = (x_coords[0] + x_coords[1]) / 2
    mid_y = (y_coords[0] + y_coords[1]) / 2
    ax.text(
      mid_x, mid_y, str(edge_no), fontsize=15, ha="center", va="center", color="black"
    )

  # Plot nodes
  node_coords_list = list(brep.nodes)
  node_to_index = {coord: i for i, coord in enumerate(node_coords_list)}

  for node_coord in brep.nodes:
    ax.plot(
      node_coord[0],
      node_coord[1],
      marker="o",
      color="red",
    )
    node_index = node_to_index[node_coord]
    ax.text(
      node_coord[0],
      node_coord[1],
      str(node_index),
      fontsize=15,
      ha="left",
      va="bottom",
      color="red",
    )

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_aspect("equal", adjustable="box")
  return ax


# plot settings for high-quality figures
high_res_plot_settings = {
  "figure.dpi": 350,
  "savefig.dpi": 350,
  "font.family": "Times New Roman",
  "font.size": 24,
  "axes.labelsize": 18,
  "axes.titlesize": 18,
  "xtick.labelsize": 18,
  "ytick.labelsize": 18,
  "legend.fontsize": 18,
  "text.usetex": True,
  "lines.linewidth": 1.5,
  "axes.linewidth": 1.0,
  "grid.linestyle": ":",
  "grid.linewidth": 0.5,
  "savefig.bbox": "tight",
  "savefig.format": "pdf",
}

# default plot settings for everyday plots
default_plot_settings = {
  "figure.dpi": 100,
  "savefig.dpi": 100,
  "font.family": "Times New Roman",
  "font.size": 10,
  "axes.labelsize": 12,
  "axes.titlesize": 14,
  "xtick.labelsize": 10,
  "ytick.labelsize": 10,
  "legend.fontsize": 10,
  "text.usetex": False,
  "lines.linewidth": 1.0,
  "axes.linewidth": 0.8,
  "grid.linestyle": "--",
  "grid.linewidth": 0.5,
  "savefig.bbox": "tight",
  "savefig.format": "png",
}

# custom colormap for fluid topology optimization
_blue = np.array([14, 135, 204]) / 255
_red = np.array([152, 0, 1]) / 255
_white = np.array([255, 255, 255]) / 255

_colors = np.vstack([_blue, _white, _red])
fluid_cmap = LinearSegmentedColormap.from_list("struct_fluid", _colors)

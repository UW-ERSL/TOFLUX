"""Simple geometry reader and processor."""

import json
import numpy as np
import shapely

class BrepGeometry:
  """Handles the reading and processing of BREP geometry files."""

  def __init__(self, brep_file: str):
    """Initializes the BrepGeometry with a file path.

    Attributes:
      brep_file: The path to the boundary representation file (json).
    """
    self.geom_file = brep_file

    with open(brep_file, "r") as f:
      self.geometry = shapely.from_geojson(f.read())
    self.edges, self.nodes = self._process_geometry()

  def _get_edges_and_nodes(self, geom: shapely.geometry.base.BaseGeometry):
    """Extracts all edges and unique nodes from a shapely geometry object.

    Args:
      geom: A shapely geometry object.

    Returns:
      A tuple containing:
        - edges: A list of tuples, where each tuple represents an edge defined by two
          coordinate points. E.g., [((x1, y1), (x2, y2)), ...].
        - nodes: A set of unique coordinate points (tuples).
    """
    edges = []
    nodes = set()
    geom_type = geom.geom_type

    # Base cases for geometries with coordinates
    if geom_type in ("LineString", "LinearRing"):
      coords = list(geom.coords)
      for i in range(len(coords) - 1):
        edges.append((coords[i], coords[i + 1]))
        nodes.add(coords[i])
        nodes.add(coords[i + 1])
      return edges, nodes

    if geom_type == "Polygon":
      # Add edges and nodes from the exterior ring
      exterior_edges, exterior_nodes = self._get_edges_and_nodes(geom.exterior)
      edges.extend(exterior_edges)
      nodes.update(exterior_nodes)
      # Add edges and nodes from all interior rings (holes)
      for interior in geom.interiors:
        interior_edges, interior_nodes = self._get_edges_and_nodes(interior)
        edges.extend(interior_edges)
        nodes.update(interior_nodes)
      return edges, nodes

    # Recursive cases for collections of geometries
    if hasattr(geom, "geoms"):  # Handles Multi-types and GeometryCollection
      for sub_geom in geom.geoms:
        sub_edges, sub_nodes = self._get_edges_and_nodes(sub_geom)
        edges.extend(sub_edges)
        nodes.update(sub_nodes)
      return edges, nodes

    return edges, nodes

  def _process_geometry(self):
    """Processes the geometry to extract edges and nodes."""

    with open(self.geom_file, "r") as f:
      geojson_data = json.load(f)
    edges = {}
    nodes = set()
    ctr = 0
    if geojson_data.get("type") == "FeatureCollection":
      for feature in geojson_data["features"]:
        geom = shapely.geometry.shape(feature["geometry"])
        feature_edges, feature_nodes = self._get_edges_and_nodes(geom)
        for edge in feature_edges:
          edges[ctr] = np.array(edge)
          ctr += 1
        nodes.update(feature_nodes)
    return edges, nodes

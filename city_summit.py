import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import pyproj as proj
import streamlit as st
from affine import Affine
from matplotlib.colors import ListedColormap
from overturemaestro import convert_geometry_to_geodataframe, convert_geometry_to_parquet, geocode_to_geometry
from pypalettes import load_cmap
from rasterio.features import MergeAlg, rasterize
from rich.progress import track
from shapely import affinity
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from streamlit.delta_generator import DeltaGenerator


def get_aligned_geometry(geometry: BaseGeometry, skip_rotation: bool = False, skip_projection: bool = False) -> float:
    centroid = geometry.centroid
    if not skip_projection:
        # assuming you're using WGS84 geographic
        crs_wgs = proj.Proj("epsg:4326")
        cust = proj.Proj(
            f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +datum=WGS84 +units=m"
        )
        project = proj.Transformer.from_proj(
            crs_wgs, cust, always_xy=True).transform

        geom_proj = transform(project, geometry)
    else:
        x, y = centroid.coords[0]
        geom_proj = affinity.translate(geometry, xoff=-x, yoff=-y)

    if skip_rotation:
        return geom_proj

    coords = list(geom_proj.minimum_rotated_rectangle.exterior.coords)

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        angle = math.degrees(
            math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        )  # https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-points
        rotated_geometry = affinity.rotate(
            geom_proj, angle=-angle, origin="centroid")

        minx, miny, maxx, maxy = rotated_geometry.bounds
        width = maxx - minx
        height = maxy - miny

        if round(height, 2) > round(width, 2):
            continue

        if abs(round(maxy, 2)) > abs(round(miny, 2)):
            continue

        return rotated_geometry

    raise RuntimeError("Rotation not found")

@st.cache_data(show_spinner=False, persist="disk", ttl=24 * 3600)
def download_overturemaps_data(location: str) -> gpd.GeoDataFrame:
    buildings_path = convert_geometry_to_parquet(
        "buildings",
        "building",
        geocode_to_geometry(location),
        # columns_to_download=["id", "geometry"],
    )
    buildings = convert_geometry_to_geodataframe(
        "buildings",
        "building",
        geocode_to_geometry(location),
        # columns_to_download=["id", "geometry"],
    )
    buildings_path.unlink()
    return buildings


@st.cache_data(show_spinner=False, persist="disk", ttl=24 * 3600)
def get_aligned_buildings(
    _st_container: DeltaGenerator, location: str, skip_rotation: bool = False, sample_size: Optional[int] = None, use_utm_projection: bool = True
) -> gpd.GeoDataFrame:
    with _st_container:
        with st.spinner('Downloading buildings from Overture Maps'):
            buildings = convert_geometry_to_geodataframe(
                "buildings",
                "building",
                geocode_to_geometry(location),
                # columns_to_download=["id", "geometry"],
            )
        
        if use_utm_projection:
            with st.spinner('Projecting buildings to UTM CRS'):
                utm_crs = buildings.estimate_utm_crs()
                buildings = buildings.to_crs(utm_crs)

    if not sample_size:
        sample_size = len(buildings)

    aligned_geometries = []
    with ProcessPoolExecutor() as ex:
        fn = partial(get_aligned_geometry,
                     skip_projection=use_utm_projection, skip_rotation=skip_rotation)
        with _st_container:
            current_progress = 0.0
            step_size = 1 / sample_size
            bar = st.progress(value=current_progress,
                              text="Aligning buildings")
            for aligned_geometry in ex.map(fn, buildings.geometry.sample(n=sample_size), chunksize=100):
                aligned_geometries.append(aligned_geometry)
                current_progress += step_size
                bar.progress(value=current_progress, text="Aligning buildings")

            bar.empty()

    return gpd.GeoDataFrame(geometry=aligned_geometries)


def generate_plotly_figure(
    title: str,
    canvas: np.ndarray,
    cmap: ListedColormap,
    camera_eye: tuple = (0.8, 0.8, 1.5),
) -> go.Figure:
    plotly_cmap = list(
        map(
            lambda c: f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})",
            cmap.colors[:, :3],
        )
    )

    y_ratio = canvas.shape[0] / canvas.shape[1]

    z = np.log(canvas)

    x_eye, y_eye, z_eye = camera_eye

    fig = go.Figure(data=[go.Surface(z=z, colorscale=plotly_cmap)])
    fig.update_traces(
        showscale=False,
        contours_z=dict(show=True, usecolormap=True, project_z=False),
    )
    fig.update_scenes(aspectratio=dict(x=1, z=1, y=y_ratio))
    fig.update_layout(
        title=dict(text=f"<br>{title}", font=dict(size=20)),
        autosize=True,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        margin=dict(l=5, r=5, b=5, t=5),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(
            visible=False), zaxis=dict(visible=False, range=[0.01, None])),
    )

    return fig


def get_city_summit(st_container: DeltaGenerator, city: str, resolution: int, skip_rotation: bool, palette_name: str) -> go.Figure:
    loaded_geometries = get_aligned_buildings(
            _st_container=st_container,
            location=city, skip_rotation=skip_rotation)

    # if skip_rotation:
    #     loaded_geometries_path = Path(
    #         f"files/buildings/{city.lower()}/aligned.parquet")
    # else:
    #     loaded_geometries_path = Path(
    #         f"files/buildings/{city.lower()}/rotated.parquet")

    # if not loaded_geometries_path.exists():
    #     loaded_geometries = get_aligned_buildings(
    #         st_container=st_container,
    #         location=city, skip_rotation=skip_rotation)
    #     loaded_geometries_path.parent.mkdir(exist_ok=True, parents=True)
    #     loaded_geometries.to_parquet(loaded_geometries_path)
    # else:
    #     with st_container:
    #         with st.spinner('Reading cached buildings'):
    #             loaded_geometries = gpd.read_parquet(loaded_geometries_path)

    gs = loaded_geometries.geometry
    minx, miny, maxx, maxy = gs.total_bounds

    canvas_width = int(np.ceil(maxx - minx)) * resolution
    canvas_height = int(np.ceil(maxy - miny)) * resolution

    with st_container:
        with st.spinner('Stacking (rasterizing) buildings'):
            canvas = rasterize(
                shapes=gs,
                fill=1,
                out_shape=(canvas_height + 4, canvas_width + 4),
                merge_alg=MergeAlg.add,
                transform=(Affine.translation(xoff=minx - 2, yoff=miny - 2)
                           * Affine.scale(1 / resolution)),
            )

    canvas = np.flipud(canvas)

    original_cm = load_cmap(palette_name, cmap_type="continuous")
    newcolors = original_cm(np.linspace(0, 1, 256))
    newcolors[:1, :] = np.array([0, 0, 0, 1])
    newcmp = ListedColormap(newcolors)

    with st_container:
        with st.spinner('Generating 3d visualization'):
            return generate_plotly_figure(city, canvas, newcmp)

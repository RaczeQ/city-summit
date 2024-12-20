import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import tempfile
from typing import Optional

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from affine import Affine
from matplotlib.colors import ListedColormap
from overturemaestro import (
    convert_geometry_to_parquet,
    geocode_to_geometry,
)
from pypalettes import load_cmap
from rasterio.features import MergeAlg, rasterize
import pyproj as proj
from shapely import affinity
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform
from streamlit.delta_generator import DeltaGenerator

BATCH_SIZE = 10_000
SAVE_DIRECTORY = Path("files/buildings")


def calculate_area(geometry: BaseGeometry) -> float:
    if isinstance(geometry, BaseMultipartGeometry):
        return sum(calculate_area(g) for g in geometry.geoms)

    centroid = geometry.centroid
    crs_wgs = proj.Proj("epsg:4326")
    cust = proj.Proj(
        f"+proj=laea +lat_0={centroid.y} +lon_0={centroid.x} +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=km +no_defs"
    )
    project = proj.Transformer.from_proj(crs_wgs, cust, always_xy=True).transform

    geom_proj = transform(project, geometry)

    return geom_proj.area


def download_overturemaps_data(location: str) -> gpd.GeoDataFrame:
    buildings_path = convert_geometry_to_parquet(
        "buildings",
        "building",
        geocode_to_geometry(location),
        max_workers=1,
        result_file_path=SAVE_DIRECTORY / location.lower() / "raw_data.parquet",
        # columns_to_download=["id", "geometry"],
        verbosity_mode="silent", 
    )
    return buildings_path


def get_cached_available_cities() -> list[str]:
    cached_cities = []
    for file_path in SAVE_DIRECTORY.glob("**/raw_data.parquet"):
        cached_cities.append(file_path.parts[-2])

    return cached_cities


def translate_geometry(geometry) -> BaseGeometry:
    return affinity.translate(
        geometry, xoff=-geometry.centroid.x, yoff=-geometry.centroid.y
    )


def rotate_geometry(geometry) -> BaseGeometry:
    coords = list(geometry.minimum_rotated_rectangle.exterior.coords)

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        angle = math.degrees(
            math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        )  # https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-points
        rotated_geometry = affinity.rotate(geometry, angle=-angle, origin="centroid")

        minx, miny, maxx, maxy = rotated_geometry.bounds
        width = maxx - minx
        height = maxy - miny

        if round(height, 2) > round(width, 2):
            continue

        if abs(round(maxy, 2)) > abs(round(miny, 2)):
            continue

        return rotated_geometry

    raise RuntimeError("Rotation not found")


def rasterize_to_canvas(geometries, total_bounds, resolution) -> np.ndarray:
    minx, miny, maxx, maxy = total_bounds

    canvas_width = int(np.ceil(maxx - minx)) * resolution
    canvas_height = int(np.ceil(maxy - miny)) * resolution

    canvas = rasterize(
        shapes=geometries,
        fill=0,
        out_shape=(canvas_height + 4, canvas_width + 4),
        merge_alg=MergeAlg.add,
        transform=(
            Affine.translation(xoff=minx - 2, yoff=miny - 2)
            * Affine.scale(1 / resolution)
        ),
    )

    return np.flipud(canvas)

def get_total_no_buildings(location: str) -> int:
    buildings_path = download_overturemaps_data(location)
    return pq.ParquetFile(buildings_path).metadata.num_rows

def get_buildings_heightmap(
    _st_container: DeltaGenerator,
    location: str,
    skip_rotation: bool = False,
    resolution: int = 1,
) -> tuple[np.ndarray, int]:
    with _st_container, st.spinner("Downloading buildings from Overture Maps"):
        buildings_path = download_overturemaps_data(location)

    with _st_container, ProcessPoolExecutor() as ex:
        final_canvas = None
        saved_prepared_geometries = []
        total_bounds = None

        rotate = not skip_rotation

        raw_file = pq.ParquetFile(buildings_path)
        total_rows = raw_file.metadata.num_rows
        total_batches = np.ceil(total_rows / BATCH_SIZE)

        current_progress = 0.0
        step_size = 1 / total_batches
        bar = st.progress(value=current_progress, text="Aligning buildings")

        for idx, batch in enumerate(raw_file.iter_batches(batch_size=BATCH_SIZE)):
            saved_aligned_buildings_path = (
                SAVE_DIRECTORY / location.lower() / "aligned" / f"{idx}.parquet"
            )
            saved_rotated_buildings_path = (
                SAVE_DIRECTORY / location.lower() / "rotated" / f"{idx}.parquet"
            )

            if saved_aligned_buildings_path.exists():
                gdf = gpd.read_parquet(saved_aligned_buildings_path)
            else:
                gdf = gpd.GeoDataFrame.from_arrow(batch).set_crs(4326)
                gdf = gdf.to_crs(gdf.estimate_utm_crs())

                gdf["geometry"] = gpd.GeoSeries(
                    ex.map(translate_geometry, gdf["geometry"], chunksize=100)
                )
                saved_aligned_buildings_path.parent.mkdir(exist_ok=True, parents=True)
                gdf.to_parquet(saved_aligned_buildings_path)

            if rotate:
                if saved_rotated_buildings_path.exists():
                    gdf = gpd.read_parquet(saved_rotated_buildings_path)
                else:
                    gdf["geometry"] = gpd.GeoSeries(
                        ex.map(rotate_geometry, gdf["geometry"], chunksize=100)
                    )
                    saved_rotated_buildings_path.parent.mkdir(
                        exist_ok=True, parents=True
                    )
                    gdf.to_parquet(saved_rotated_buildings_path)

                saved_prepared_geometries.append(saved_rotated_buildings_path)
            else:
                saved_prepared_geometries.append(saved_aligned_buildings_path)

            current_progress += step_size
            bar.progress(value=current_progress, text="Aligning buildings")

            batch_total_bounds = gdf["geometry"].total_bounds

            if total_bounds is None:
                total_bounds = batch_total_bounds
                continue

            if batch_total_bounds[0] < total_bounds[0]:
                total_bounds[0] = batch_total_bounds[0]
            if batch_total_bounds[1] < total_bounds[1]:
                total_bounds[1] = batch_total_bounds[1]
            if batch_total_bounds[2] > total_bounds[2]:
                total_bounds[2] = batch_total_bounds[2]
            if batch_total_bounds[3] > total_bounds[3]:
                total_bounds[3] = batch_total_bounds[3]

        bar.empty()

        current_progress = 0.0
        bar = st.progress(
            value=current_progress, text="Stacking (rasterizing) buildings"
        )

        for file_path in saved_prepared_geometries:
            gdf = gpd.read_parquet(file_path)
            canvas = rasterize_to_canvas(gdf["geometry"], total_bounds, resolution)
            if final_canvas is None:
                final_canvas = canvas
            else:
                final_canvas += canvas
            current_progress += step_size
            bar.progress(
                value=current_progress, text="Stacking (rasterizing) buildings"
            )

        bar.empty()

    return final_canvas


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

    with np.errstate(divide="ignore"):
        z = np.log(canvas)
        z[np.isneginf(z)] = -1

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
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[-0.99, None]),
        ),
    )

    return fig


def get_city_summit(
    st_container: DeltaGenerator,
    city: str,
    resolution: int,
    skip_rotation: bool,
    palette_name: str,
    reverse_palette: bool,
) -> go.Figure:
    canvas_file_type = "aligned" if skip_rotation else "rotated"
    saved_canvas_path = (
        SAVE_DIRECTORY / city.lower() / f"{canvas_file_type}_{resolution}.npy"
    )

    if not saved_canvas_path.exists():
        canvas = get_buildings_heightmap(
            _st_container=st_container,
            location=city.lower(),
            skip_rotation=skip_rotation,
            resolution=resolution,
        )
        np.save(saved_canvas_path, canvas)
    else:
        canvas = np.load(saved_canvas_path)

    original_cm = load_cmap(
        palette_name, cmap_type="continuous", reverse=reverse_palette
    )
    newcolors = original_cm(np.linspace(0, 1, 256))
    newcolors[:1, :] = np.array([0, 0, 0, 1])
    newcmp = ListedColormap(newcolors)

    with st_container, st.spinner("Generating 3d visualization"):
        total_no_buildings = get_total_no_buildings(city.lower())
        return generate_plotly_figure(f"{city} ({total_no_buildings} buildings)", canvas, newcmp)

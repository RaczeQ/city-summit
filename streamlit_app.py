import random
from typing import Optional

import geopandas as gpd
import plotly.express as px
import streamlit as st
from overturemaestro import geocode_to_geometry
from overturemaestro._exceptions import QueryNotGeocodedError
from streamlit_folium import st_folium

from city_summit import calculate_area, get_cached_available_cities, get_city_summit

st.set_page_config(
    page_title="City Summit generator",
    page_icon=":material/elevation:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

MAX_CITY_AREA_KM2 = 50_000

add_new_city_prompt = "Add new city..."


def get_selected_city() -> Optional[str]:
    return st.session_state.get("selected_city")


def is_city_selected() -> bool:
    return get_selected_city() is not None


def select_city(city: str):
    if city == add_new_city_prompt:
        st.session_state["selected_city"] = None
    st.session_state["selected_city"] = city


def is_executed():
    return st.session_state.get("executed") is not None


def set_executed():
    st.session_state["executed"] = True


st.title("City Summit 🏙️🗻", anchor=False)

tab1, tab2 = st.tabs(["Generate", "About"])

container1 = tab1.container()
container2 = tab1.container(border=True)

disable_button = False

with container1:
    st.write("Generate a summit for the city of your choosing!")

    cached_cities = get_cached_available_cities()

    new_city_name = None
    select_options = [add_new_city_prompt, *sorted(cached_cities)]
    selected_city_name = st.sidebar.selectbox(
        "Select the city",
        options=select_options,
        index=(
            select_options.index(st.session_state.get("city_select_option"))
            if "city_select_option" in st.session_state
            else (random.randint(1, len(cached_cities)) if cached_cities else 0)
        ),
        help="Will be geocoded using Nominatim service. Bigger cities / regions will crash the runtime!",
        key="city_select_option",
    )
    if selected_city_name == add_new_city_prompt:
        new_city_name = st.sidebar.text_input(
            "New city to geocode",
            help="Will be geocoded using Nominatim service. Bigger cities / regions will crash the runtime!",
        )

    selected_city_name = (
        new_city_name
        if selected_city_name == add_new_city_prompt
        else selected_city_name
    )

    select_city(selected_city_name)

    if not disable_button and not selected_city_name:
        disable_button = True

    if selected_city_name:
        try:
            geocoded_geometry = geocode_to_geometry(selected_city_name)
            folium_map = gpd.GeoSeries([geocoded_geometry], crs=4326).explore(
                tiles="CartoDB Voyager"
            )
            with st.sidebar:
                st_folium(
                    folium_map,
                    height=200,
                    use_container_width=True,
                    returned_objects=[],
                )
            if calculate_area(geocoded_geometry) > MAX_CITY_AREA_KM2:
                st.sidebar.error("Area is too big.", icon="🐋")
                disable_button = True
        except QueryNotGeocodedError:
            st.sidebar.error("Query not geocoded.", icon="🌍")
            disable_button = True
        except Exception:
            st.sidebar.error("Unknown geocoding error", icon="🌐")
            disable_button = True

    with st.sidebar.container():
        col1, col2 = st.columns([0.6, 0.4], vertical_alignment="bottom")
        color_palette = col1.text_input(
            "Palette",
            value="ag_Sunset",
            help="For more palettes look here https://python-graph-gallery.com/color-palette-finder/",
        )
        reverse_palette = col2.checkbox(
            "Reversed",
            value=False,
        )

    if not disable_button and not color_palette:
        disable_button = True

    resolution = st.sidebar.selectbox(
        "Resolution",
        options=(1, 2, 3, 4, 5),
        help="Increasing the resolution for bigger cities might crash the execution.",
    )
    rotate_buildings = st.sidebar.checkbox(
        "Rotate buildings",
        value=True,
        help="Rotation will align building in similar orientation angle-wise.",
    )

    if not is_executed():
        st.caption("⬅️ Set parameters in the sidebar.")


def render_summit():
    set_executed()
    with container2:
        fig = get_city_summit(
            st_container=container2,
            city=selected_city_name,
            resolution=resolution,
            skip_rotation=not rotate_buildings,
            palette_name=color_palette,
            reverse_palette=reverse_palette,
        )
        with st.spinner("Generating 3d visualization"):
            st.plotly_chart(fig, use_container_width=True)


with container1:
    st.sidebar.button("Generate!", on_click=render_summit, disabled=disable_button)

tab2.subheader("Project description", anchor=False, divider=True)
tab2.markdown("""
**City Summit** 🏙️🗻 is a visualization project focused on exploring the geometries of buildings from a given city.
""")
with tab2.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col3:
        st.write(" ")
    with col2:
        st.caption("Example summit visualization for the city of Wrocław, Poland.")
        st.image("images/wro_summit_transparent.png")
tab2.markdown("""
Created visualizations represent stacked layers made from existing buildings in a given city.
              
The name of the project refers to the shape of the mountain, which is created by layering the buildings on top of each other in this way.
""")
tab2.subheader("How does it work?", divider=True, anchor=False)
tab2.markdown("""
To generate the visualization, app downloads buildings for a given city from the [Overture Maps](https://overturemaps.org/) dataset.

After downloading the buildings, all geometries are projected to the Universal Transverse Mercator (UTM) Coordinate Reference System and their vertices are translated, so that the centroid of a building is at a point (0,0).
              
Optionally, the buildings are also rotated around the centroid, so that the minimum rotated rectangle is equal to the bounding box of a geometry. This operation "straightens" the buildings so that the edges of buildings are aligned with each other.
              
Next, the rasterization operation is applied to a whole dataset. All geometries are stacked on top of each other creating a heightmap.
              
Last operation is plotting the created hightmap as a 3D surface plot. The heighmap data is scaled logarithmically, to accentuate bigger buildings.
""")
tab2.subheader("Used tools", divider=True, anchor=False)
tab2.markdown("""
This website is written as a Streamlit app and utilizes the following libraries:
              
- [OvertureMaestro](https://github.com/kraina-ai/overturemaestro) - for downloading Overture Maps data for a given city and for geocoding the string to a geometry.
- [GeoPandas](https://github.com/geopandas/geopandas) - for geospatial operations on a dataset of downloaded buildings and I/O.
- [Shapely](https://github.com/shapely/shapely) - for rotations and translations of a building geometry.
- [Rasterio](https://github.com/rasterio/rasterio) - for rastering the buildings dataset into a heightmap.
- [Plotly](https://github.com/plotly/plotly.py) - for displaying 3D surface plot using heightmap data.
- [NumPy](https://github.com/numpy/numpy) - for 2D array (heightmap) manipulation.
- [PyPalettes](https://github.com/JosephBARBIERDARNAL/pypalettes) - for easy access to many palletes.
- [Matplotlib](https://github.com/matplotlib/matplotlib) - for transforming palettes.
""")
tab2.divider()
tab2.markdown("""
**Hi**! I'm Kamil 👋🏻
              
You can see my other projects here: https://kamilraczycki.com/projects/
""")

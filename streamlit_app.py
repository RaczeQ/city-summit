import streamlit as st
import plotly.express as px

from city_summit import get_city_summit

st.set_page_config(page_title="City Summit generator", page_icon=":material/elevation:", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

st.title("City Summit 🏙️🗻", anchor=False)

tab1, tab2 = st.tabs(["Generate", "About"])

container1 = tab1.container()
container2 = tab1.container(border=True)

with container1:
    st.write(
        "Generate a summit for the city of your choosing!"
    )

    city_name = st.sidebar.text_input(
        "City to geocode", help="Will be geocoded using Nominatim service")
    color_palette = st.sidebar.text_input('Palette', value="ag_Sunset",
                                          help="For more palettes look here https://python-graph-gallery.com/color-palette-finder/")
    resolution = st.sidebar.selectbox("Resolution", options=(
        1, 2, 3, 4, 5), help="Increasing the resolution for bigger cities might crash the execution.")
    rotate_buildings = st.sidebar.checkbox("Rotate buildings", value=True,
                                           help="Rotation will align building in similar orientation angle-wise.")

    if not city_name:
        st.caption(
            "Set parameters in the sidebar."
        )


def render_summit():
    with container2:
        fig = get_city_summit(st_container=container2, city=city_name, resolution=resolution,
                              skip_rotation=not rotate_buildings, palette_name=color_palette)
        with st.spinner('Generating 3d visualization'):
            st.plotly_chart(fig, use_container_width=True)


with container1:
    disable_button = not city_name or not color_palette
    st.sidebar.button('Generate!', on_click=render_summit,
                      disabled=disable_button)

tab2.subheader("Project description", anchor=False, divider=True)
tab2.markdown("""
**City Summit** 🏙️🗻 is a visualization project focused on exploring the geometries of buildings from a given city.
""")
with tab2.container(border=True):
    st.caption("Example summit visualization for the city of Wrocław, Poland.")
    st.image("images/wro_summit_transparent.png")
tab2.markdown("""
Created visualizations represent stacked layers made from existing buildings in a given city.
              
The name of the project refers to the shape of the gable, which is created by layering the buildings on top of each other in this way.
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
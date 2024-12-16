import streamlit as st
import plotly.express as px

from city_summit import get_city_summit

container1 = st.container()
container2 = st.container(border=True)

with container1:
    st.title("City Summit 🏙️🗻")
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

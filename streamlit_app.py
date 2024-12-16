import time
import streamlit as st
import plotly.express as px

st.title("City Summit 🏙️🗻")
st.write(
    "Generate a summit for the city of your choosing!"
)

city_name = st.text_input('City')
resolution = st.slider("Resolution", min_value=1,
                       max_value=10, value=1, step=1)
rotate_buildings = st.checkbox("Rotate buildings", value=True)

if st.button('Generate!'):
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success("Done!")

    df = px.data.iris()
    fig = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        size="petal_length",
        hover_data=["petal_width"],
    )

    event = st.plotly_chart(fig, key="iris", on_select="rerun")

import geopandas as gpd
import pandas as pd
import streamlit as st

from explore_data import show_explore_data_page
from predict_page import load_predict_page

# the theme of the web app is defined in the config.toml file
st.set_page_config(
    page_title="Covid-19 App",
    layout="wide",
    initial_sidebar_state="expanded",
)

concelhos_format = gpd.read_file(
    r"Geographic_Info/concelhos_shape_file/concelhos_format.shp"
)


@st.cache
def load_all_necessary_data():
    # Upload of the Raw Data and then All_Data_Reformed for the SOM Analysis and also the lat_long_info of concelhos
    raw_data = pd.read_csv(r"data/raw_data_Covid19.csv", encoding="ISO-8859-1")
    raw_data_Covid19 = raw_data.copy()
    raw_data_Covid19.set_index("Concelhos", inplace=True)
    # upload of the data with all the wanted indicators
    data_all = pd.read_csv(
        r"data/data_thesis.csv",
        encoding="UTF-8-SIG",
        usecols=lambda c: not c.startswith("Unnamed:"),
    )
    all_data = data_all.copy()
    # just because the incidences column calculation sometimes returns numbers full of decimals
    all_data.round(4)
    all_data.set_index("Concelhos", inplace=True)
    # importing the necessary incidence_14_days data
    Incidences_14days = pd.read_csv(r"data/inc_14_fixed.csv", encoding="utf-8")
    inc_14days = Incidences_14days.copy()
    inc_14days.round(4)
    inc_14days.set_index("Concelhos", inplace=True)
    Data_incidences = inc_14days.copy()
    # importing and building the geo_other_data_concelhos
    other_data = all_data.iloc[
        :, all_data.columns.get_loc("dens_pop") : all_data.shape[1]
    ].copy()  # dens_pop is the
    geo_other_data_concelhos = pd.merge(
        concelhos_format,
        other_data,
        left_on="Concelho",
        right_on="Concelhos",
        how="left",
    ).copy()
    return raw_data_Covid19, all_data, Data_incidences, geo_other_data_concelhos


(
    raw_data_Covid19,
    all_data,
    Data_incidences,
    geo_other_data_concelhos,
) = load_all_necessary_data()


# now we just want to still make a sidebar and the other web app page, where we can explore all
# the data used for the model s training

# we can move any widge(button, selectbox, slider etc) to the sidebar by using it as a prefix
user_wanted_page = st.sidebar.selectbox(
    "Explore Covid-19 Data or Use SOM clustering as Classification Tool",
    ["Explore", "Classification Tool"],
)

if user_wanted_page == "Explore":
    show_explore_data_page(
        raw_data_Covid19, Data_incidences, all_data, geo_other_data_concelhos
    )
else:
    load_predict_page(all_data)

# Import of all the necessary libraries for the work
import logging
import pickle

import geopandas as gpd

# libraries to save and load the SOM trained models
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import cm
from matplotlib.colors import Normalize

logging.basicConfig(level="INFO")

mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
"""
Auxiliary Function to save and load objects using the pickle module
"""


def load_obj(name):
    with open("objetos/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


"""
import all the data necessary for the plotting
"""


def processing_all_needed_data(all_data, altura_do_ano):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Summer Season", ("2020-07-01", "2020-09-10")),
            ("September-October of 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of COVID-19", ("2020-10-01", "2020-12-15")),
            ("Christmas and Holiday Season", ("2020-12-15", "2021-02-06")),
        ]
    )
    # data_incidences a ter a info das incidencias e other_data dos
    # outras features de todos os concelhos nesta epoca de tempo
    data_incidences_df = all_data.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    the_other_data = all_data.iloc[
        :, all_data.columns.get_loc("2021-02-06") + 1 : all_data.shape[1]
    ].copy()
    all_data_needed = pd.merge(
        data_incidences_df, the_other_data, how="inner", on="Concelhos"
    )
    # data Standardization(mean of 0, standard deviation of 1) before applying SOM
    all_data_needed_st = (all_data_needed - np.mean(all_data_needed, axis=0)) / np.std(
        all_data_needed, axis=0
    )
    return all_data_needed_st


"""
import the incidence data necessary for the plotting
"""


def processing_incidence_needed_data(Data_incidences, altura_do_ano):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Summer Season", ("2020-07-01", "2020-09-10")),
            ("September-October of 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of COVID-19", ("2020-10-01", "2020-12-15")),
            ("Christmas and Holiday Season", ("2020-12-15", "2021-02-06")),
        ]
    )
    # Data standardization and preparation with respect to the time of the year chosen
    Data_incidences_standardized = (
        (Data_incidences - np.mean(Data_incidences, axis=0))
        / np.std(Data_incidences, axis=0)
    ).copy()
    Data_incidences_needed = Data_incidences_standardized.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    return Data_incidences_needed


"""
import the geodataframes necessary for the plotting
"""


def importing_lat_long_concelho_data():
    # districts_format = gpd.read_file('districts_shape_file\districts_format.shp')
    concelhos_format = gpd.read_file(
        r"Geographic_Info/concelhos_shape_file/concelhos_format.shp"
    )
    concelhos_format.set_index("Concelho", inplace=True)
    # to avoid a keyerror from the mismatch between Ponte de Sor as it s in the clustering_groups and
    # Ponte de Sôr in the concelhos_format index
    concelhos_format.index = concelhos_format.index.str.replace(
        "Ponte de Sôr", "Ponte de Sor"
    )
    concelhos_lat_long_geo_data = gpd.read_file(
        r"Geographic_Info/concelhos_lat_long_shapefile/concelhos_lat_long.shp"
    )
    concelhos_lat_long_geo_data.set_index("Concelho", inplace=True)
    return concelhos_format, concelhos_lat_long_geo_data


"""
Generic Functions that allow at any stage to have the SOM_clustering_map and SOM_clustering_grid
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def SOM_clustering_grid(all_data_needed, altura_do_ano, colors_per_neuron):
    data = all_data_needed.values
    # load and test the saved  trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    n_neurons, m_neurons = 5, 5
    # calculo de todos as posiçoes dos winning neurons para cada um dos data inputs
    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x, w_y = np.array(w_x), np.array(w_y)
    # colocar como background Matriz U de distancia, building the graphic part
    fig = plt.figure(figsize=(20, 20))
    ax = fig.gca()
    ax.set_title(
        f"U-Matrix of Output Space in {altura_do_ano}",
        fontweight="bold",
        fontsize=30,
        y=1.01,
        loc="center",
    )
    # plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.9)
    plt.imshow(
        som.distance_map().T,
        cmap="bone_r",
        alpha=0.9,
        origin="lower",
        aspect="equal",
        extent=[-1, 4, -1, 4],
    )
    cbar = plt.colorbar(
        fraction=0.046, pad=0.04, ticks=[tick for tick in np.arange(0, 1.1, 0.1)]
    )
    cbar.set_label(
        "Distance to its Neighboring Neurons", fontsize=20, labelpad=25, rotation=-90
    )
    label_names = all_data_needed.index
    # fazer plot da posição ocupada pelo winning neuron de cada um dos data inputs,cada um dos concelhos
    # Em cada ciclo ocorre o plot de cada concelho # é adicionado um numero random, para que não acham concelhos
    # a coincidir mesmo quando sao mapeados para o mesmo winning neuron
    for indice_concelho, concelho_input in enumerate(label_names):
        # plt.scatter(w_x[indice_concelho]+np.random.uniform(-0.65,0)-0.2,
        # w_y[indice_concelho]+np.random.uniform(-0.65,0)-0.2,
        # s=90, color=dic_colors_concelhos[concelho_input], label=concelho_input)
        plt.scatter(
            w_x[indice_concelho] + np.random.uniform(-0.65, 0) - 0.2,
            w_y[indice_concelho] + np.random.uniform(-0.65, 0) - 0.2,
            s=90,
            color=colors_per_neuron[(w_x[indice_concelho], w_y[indice_concelho])],
        )
    plt.xticks(np.arange(-0.5, n_neurons - 0.5, 1), np.arange(5), fontsize=12)
    plt.yticks(np.arange(-0.5, m_neurons - 0.5, 1), np.arange(5), fontsize=12)
    plt.xlabel("Neuron X Coordinate", fontsize=15, labelpad=5)
    plt.ylabel("Neuron Y Coordinate", fontsize=15, labelpad=5)
    # plt.legend(loc='upper right', ncol=4,bbox_to_anchor=(2.3, 1.22),fontsize=14, markerscale=2,borderpad=2)
    # st.pyplot(fig)  # to be able to display this image in the streamlit web app
    return fig


@st.cache(
    hash_funcs={matplotlib.figure.Figure: lambda _: None},
    suppress_st_warning=True,
    allow_output_mutation=True,
)
def SOM_clustering_map(all_data_needed, altura_do_ano, colors_per_neuron):
    # loading the previous trained SOM for this time period
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    # uploading the concelhos format geodataframe
    concelhos_format, _ = importing_lat_long_concelho_data()
    color_concelho = [
        list(colors_per_neuron[som.winner(all_data_needed.loc[concelho].values)])
        for concelho in concelhos_format.index
    ]
    # adding the correspodent color for each concelho as a new column, respecting their BMU color unit
    concelhos_format["color_concelhos"] = color_concelho
    # plotting section
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(15, 15))
    # First, Plotting Portugal and its concelhos borders, and each concelho with the corresponding color of their BMU
    concelhos_format.plot(
        ax=ax, edgecolor="black", color=concelhos_format["color_concelhos"]
    )
    # setting up the image title, axis names, ticks etc
    ax.set_title(
        f"Output Space Cartographic Map in {altura_do_ano}",
        fontweight="bold",
        fontsize=20,
        loc="center",
        y=0.97,
    )
    ax.set_xlabel("Longitude", fontsize=15, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=15, labelpad=5)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.axis("off")
    # plt.show()
    return fig


"""
Function that plots incidences for each of the neurons in one of the specific times
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_raw_incidências_per_neuron_fill_between(
    raw_data, all_data, altura_do_ano, colors_per_neuron
):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Summer Season", ("2020-07-01", "2020-09-10")),
            ("September-October of 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of COVID-19", ("2020-10-01", "2020-12-15")),
            ("Christmas and Holiday Season", ("2020-12-15", "2021-02-06")),
        ]
    )
    # upload da raw data com os new positive cases of Covid_19
    raw_data_Covid19 = raw_data.copy()
    all_data_needed = processing_all_needed_data(all_data, altura_do_ano)
    data = all_data_needed.values
    # load and test the saved trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data, all_data_needed.index)
    # bulding a dataframe with all the data incidences average daily values for each neuron in the time stage chosen
    df_data = []
    for neuron_coordinates in list(
        colors_per_neuron.keys()
    ):  # (specific neuron coordinates): [specific color]
        # have to confirm if they were mappings in this neuron, if not, continue to the next iteration of the loop, no need to plot their incidence
        if neuron_coordinates not in dic_neuron_labels.keys():
            continue
        # list of concelhos mapped to the neuron
        concelhos = list(dict(dic_neuron_labels)[neuron_coordinates].keys())
        # using the raw covid19 csv to sum all the positive cases among the group of concelhos mapped to the current neuron being analysed
        Covid19_filtrada = raw_data_Covid19.loc[concelhos, :].copy()
        data_incidences, days = [], []
        n_firstday = Covid19_filtrada.columns.get_loc(stages[altura_do_ano][0])
        n_lastday = Covid19_filtrada.columns.get_loc(stages[altura_do_ano][1])
        for firstday, lastday in zip(
            range(n_firstday - 13, n_lastday + 1, 1),
            range(n_firstday, n_lastday + 1, 1),
        ):
            Total_Cases = (
                Covid19_filtrada.loc[
                    concelhos,
                    Covid19_filtrada.columns[firstday] : Covid19_filtrada.columns[
                        lastday
                    ],
                ]
                .sum(axis=1)
                .sum()
            )
            data_incidences += [
                (
                    (
                        ((Total_Cases / Covid19_filtrada["Populacao"].sum()) * 100000)
                    ).round(decimals=4)
                )
            ]
            days += [f"{Covid19_filtrada.columns[lastday]}"]
        df_data += [[neuron_coordinates, data_incidences]]
    neuron_coords_data_incidences = pd.DataFrame(
        df_data, columns=["Neuron_coordinates", "Data_Incidences"]
    ).copy()
    fig = plt.figure(figsize=(100, 60))
    ax = fig.gca()
    ax.set_title(
        f"Average Incidences of Concelhos Mapped per Neuron in {altura_do_ano}",
        fontweight="bold",
        fontsize=90,
        y=1.01,
    )
    # contrução de df correspondente à Incidência Nacional com base na raw data de Covid inicial -- plot da incidência nacional
    Covid19 = raw_data_Covid19.copy()
    national_incidences = pd.DataFrame(
        ["Incidência Nacional"], columns=["Concelhos"]
    ).set_index("Concelhos")
    n_firstday = Covid19.columns.get_loc(stages[altura_do_ano][0])
    n_lastday = Covid19.columns.get_loc(stages[altura_do_ano][1])
    for firstday, lastday in zip(
        range(n_firstday - 13, n_lastday + 1, 1), range(n_firstday, n_lastday + 1, 1)
    ):
        Nacional_Cases = (
            Covid19.loc[:, Covid19.columns[firstday] : Covid19.columns[lastday]]
            .sum(axis=1)
            .sum()
        )
        national_incidences[f"{Covid19.columns[lastday]}"] = (
            ((Nacional_Cases / Covid19["Populacao"].sum()) * 100000)
        ).round(decimals=4)
    # double_the_national_incidences = national_incidences * 2
    ax.plot(
        days,
        national_incidences.loc["Incidência Nacional"].values.tolist(),
        color="black",
        label="Nacional Incidence",
        linewidth=40,
    )
    # ax.plot(days,double_the_national_incidences.loc['Incidência Nacional'].values.tolist(),color='black',label='Double the Nacional Incidence',linewidth=40)
    # fill the plot with green below the Incidência Nacional
    ax.fill_between(
        days,
        national_incidences.loc["Incidência Nacional"].values.tolist(),
        color="green",
        alpha=0.2,
        label="Below National Incidence",
    )
    # using the previous built dataframe to plot the data incidences for each neuron
    for neuron_coords in neuron_coords_data_incidences["Neuron_coordinates"]:
        filt = neuron_coords_data_incidences["Neuron_coordinates"] == neuron_coords
        ax.plot(
            days,
            neuron_coords_data_incidences.loc[filt, "Data_Incidences"].values[0],
            color=colors_per_neuron[neuron_coords],
            label=[neuron_coords],
            linewidth=15,
        )
        # ax.fill_between(days, neuron_coords_data_incidences.loc[filt, 'Data_Incidences'].values[0],national_incidences.loc['Incidência Nacional'].values.tolist(),
        # where = (np.array(neuron_coords_data_incidences.loc[filt, 'Data_Incidences'].values[0]) > np.array(national_incidences.loc['Incidência Nacional'].values.tolist())),
        # color = "red", alpha = 0.08)
    # Plot das Linhas de Risco Definidas pela DGS
    ax.plot(
        days,
        [240] * len(days),
        color="palegreen",
        linewidth=15,
        label="Moderate Risk",
        linestyle="--",
    )
    # ax.fill_between(days, [240]*len(days), color = "palegreen", alpha = 0.2) #fill plot with green for incidences below 240
    ax.plot(
        days,
        [480] * len(days),
        color="gold",
        linewidth=15,
        label="High Risk",
        linestyle="--",
    )
    # ax.fill_between(days, [480]*len(days), [240]*len(days), color = "gold", alpha = 0.2) #fill plot with gold for incidences between 240 and 480
    # ax.plot(days, [960]*len(days), color = "darkorange",linewidth=15,label='Very High Risk',linestyle='--')
    # ax.fill_between(days, [960]*len(days), [480]*len(days), color =  "darkorange", alpha = 0.2) #fill plot with darkorange for incidences between 960 and 480
    # setting up the image axis, labels etc
    ax.set_xlabel("Days", fontsize=80, labelpad=10)
    ax.set_ylabel(
        "Covid-19 Cum.Incidence for 14Days per 100 000 hab", fontsize=80, labelpad=15
    )
    # timearray=np.arange('2020-03-28', '2020-06-01',np.timedelta64(5,'D'), dtype='datetime64')
    timearray = days[::5]  # tick time intervals in the x axis
    plt.xticks(timearray, fontsize=50)
    plt.yticks(fontsize=50)
    ax.tick_params(axis="both", direction="inout", length=60, width=10, color="black")
    # plt.legend(bbox_to_anchor=(1.5, 1.5))
    plt.legend(loc="best", fontsize=50)
    plt.grid()
    # plt.show()
    return fig


"""
Function which plots the geospatial pattern of neurons Above Average in any of the critical times
"""


@st.cache(
    hash_funcs={matplotlib.figure.Figure: lambda _: None},
    suppress_st_warning=True,
    allow_output_mutation=True,
)
def plot_concelhos_in_risk_neurons(
    all_data_needed, altura_do_ano, dic_neurons_above_average, colors_per_neuron
):
    # loading the previous trained SOM for this time period
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    # uploading the concelhos format geodataframe
    concelhos_format, _ = importing_lat_long_concelho_data()
    # dic_neurons_above_average = load_obj('dic_neurons_above_average')
    high_risk_neurons = dic_neurons_above_average[altura_do_ano]
    color_concelho = []
    for concelho in concelhos_format.index:
        concelho_neuron = som.winner(all_data_needed.loc[concelho].values)
        if concelho_neuron in high_risk_neurons:
            color_concelho += [list(colors_per_neuron[concelho_neuron])]
        else:
            color_concelho += [[1, 1, 1]]
    # adding the correspodent color for each concelho as a new column, atributing white to all the concelhos not mapped to high_risk_neurons
    # to the others, the correspodent color of the high risk BMU to which they were mapped
    concelhos_format["color_concelhos"] = color_concelho
    # plotting section
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(15, 15))
    # First, Plotting Portugal and its concelhos borders, and each concelho with the corresponding color of their BMU
    concelhos_format.plot(
        ax=ax, edgecolor="black", color=concelhos_format["color_concelhos"]
    )
    # setting up the image title, axis names, ticks etc
    ax.set_title(
        f"Cartographic Map of High Risk Neurons in {altura_do_ano}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
        loc="center",
    )
    ax.set_xlabel("Longitude", fontsize=15, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=15, labelpad=5)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.legend(bbox_to_anchor=(1.50, 1.01),loc = "upper right",fontsize=8)
    plt.axis("off")
    # plt.show()
    return fig


"""
Function capable of computing the heat_map for all the features in the other data, for all the concelhos mapped to each of 
the high and low risk incidence neurons
"""
feature_names_list = [
    "Population Density",
    "Deprivation Score",
    "Youth Population",
    "Eldery Population",
    "Jobs in the Primary Sector",
    "Jobs in the Secondary Sector",
    "Jobs in the Tertiary Sector",
    "People Needing State Benefits",
    "Number of Schools",
]

"""Function Which which Plots the quantiles distribution for any of the features in Portugal's Mainland, Legend in colorbar format"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_feature_distribution_in_Portugal_cmap(
    feature_to_plot, geo_other_data_concelhos
):
    dic_feature_translation = {
        full_feature_name: abreviation
        for abreviation, full_feature_name in zip(
            geo_other_data_concelhos.columns[2:], feature_names_list
        )
    }
    fig, gax = plt.subplots(figsize=(5, 5))
    # Plotting Portugal concelho borders and pass 'feature_to_plot' as the feature to be based the color adjustment of the concelhos
    # geo_other_data_concelhos.plot(ax = gax, edgecolor='black', column='dens_pop', legend=True, cmap='RdYlGn')
    plot_img = geo_other_data_concelhos.plot(
        ax=gax,
        edgecolor="black",
        column=f"{dic_feature_translation[feature_to_plot]}",
        cmap="coolwarm",
        scheme="quantiles",
    )
    # buldinng the cmap manually to not use the quantiles
    norm = Normalize(
        vmin=geo_other_data_concelhos[dic_feature_translation[feature_to_plot]].min(),
        vmax=geo_other_data_concelhos[dic_feature_translation[feature_to_plot]].max(),
    )
    n_cmap = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    n_cmap.set_array([])
    plot_img.get_figure().colorbar(n_cmap)
    # labeling the axis, giving a title to the plot etc
    gax.set_xlabel("Longitude", fontsize=15)
    gax.set_ylabel("Latitude", fontsize=15)
    # gax.set_title(f"Portugal {feature_to_plot}", fontsize=10, y=1.01, loc="center")
    # Kill the spines...
    gax.spines["top"].set_visible(False)
    gax.spines["right"].set_visible(False)
    plt.axis("off")
    # plt.show()
    return fig

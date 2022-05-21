# Importe de Todas as Bibliotecas Necessárias
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from functions_explore_page import importing_lat_long_concelho_data


# auxiliary function used by new_input_assignment function
def contemQ(l1, el):
    if l1 == []:
        return False
    return next((True for element in l1 if element == el), False)


""" 
Function which predicts the neuron in which the new input would be mapped in all of the timeframes
"""


def forecast_new_data_input_mappings(all_data, new_data_name):
    Concelho_Clusters = pd.DataFrame(["New_Input"], columns=["Concelho"]).set_index(
        "Concelho"
    )
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Summer Season", ("2020-07-01", "2020-09-10")),
            ("September-October of 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of COVID-19", ("2020-10-01", "2020-12-15")),
            ("Christmas and Holiday Season", ("2020-12-15", "2021-02-06")),
        ]
    )
    for epoca in stages:
        # calculo de neuron para um concelho em específico, com new_data_input a ter a info das incidencias e da other data
        # desse nesta epoca de tempo
        # data Standardization(mean of 0, standard deviation of 1) before splicing the data and then applying SOM
        all_data_standardized = (
            (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
        ).copy()
        new_data_input = all_data_standardized.loc[new_data_name].copy()
        new_data_input_incidences = new_data_input.loc[
            stages[epoca][0] : stages[epoca][1]
        ].copy()
        new_data_input_other_data = new_data_input.loc["dens_pop":].copy()
        # new_data_input = new_data_input_incidences.append(
        # new_data_input_other_data
        # ).copy()
        new_data_input = pd.concat(
            [new_data_input_incidences, new_data_input_other_data]
        ).copy()
        new_data = new_data_input.values
        som = joblib.load(f"Trained_Models/SOM_{epoca}")
        neuron_predicted = som.winner(new_data)
        Concelho_Clusters[f" Cluster in {epoca} "] = [neuron_predicted]
    return Concelho_Clusters


"""
Classifies a new input, by returning the neuron coords of the neuron in which it was mapped and the other concelhos already mapped to that neuron
"""


def classify(all_data, concelho, altura_do_ano):
    """Classifies the new_input to one of the neurons defined
    using the method labels_map.
    Returns a list of the "concelhos" assigned to the same neuron and the
    neurons coordinates
    """
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Summer Season", ("2020-07-01", "2020-09-10")),
            ("September-October of 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of COVID-19", ("2020-10-01", "2020-12-15")),
            ("Christmas and Holiday Season", ("2020-12-15", "2021-02-06")),
        ]
    )

    # data Standardization(mean of 0, standard deviation of 1) before applying SOM
    all_data_standardized = (
        (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
    ).copy()
    # here the new_data_input corresponds to Lagoa, but ideally it would be other random concelho
    new_data_input = all_data_standardized.loc[concelho].copy()
    new_data_input_incidences = new_data_input.loc[
        stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    new_data_input_other_data = new_data_input.loc["dens_pop":].copy()
    new_data_input = new_data_input_incidences.append(new_data_input_other_data).copy()
    data = new_data_input.values
    # Calculo das restantes incidências para todos os outros concelhos e other data também
    data_incidences_df = all_data_standardized.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    the_other_data = all_data_standardized.iloc[
        :,
        all_data_standardized.columns.get_loc("dens_pop") : all_data_standardized.shape[
            1
        ],
    ].copy()
    all_data_needed = pd.merge(
        data_incidences_df, the_other_data, how="inner", on="Concelhos"
    )
    data_SOM = all_data_needed.values
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data_SOM, all_data_needed.index)
    # form a dic with neuron_coordinate : [list of concelhos mapped to it]
    dic_neuron_concelhos = {}
    for neuron_coordinates in list(dic_neuron_labels.keys()):
        dic_neuron_concelhos[neuron_coordinates] = list(
            dict(dic_neuron_labels)[neuron_coordinates].keys()
        )
    coordinates_winning_neuron = som.winner(data)
    concelhos_mapped_to_same_neuron = []
    concelhos_mapped_to_same_neuron.append(
        dic_neuron_concelhos[coordinates_winning_neuron]
    )
    return coordinates_winning_neuron, concelhos_mapped_to_same_neuron


"""
Function which plots the concelhos that were mapped to the same neuron as the new input that the user wanted to classify
"""


@st.cache(
    hash_funcs={matplotlib.figure.Figure: lambda _: None},
    suppress_st_warning=True,
    allow_output_mutation=True,
)
def plot_concelhos_classified_together(
    concelhos_mapped_together, colors_per_neuron, neuron_coords, concelho_selected
):
    # import the geodataframes necessary for the plotting
    concelhos_format, _ = importing_lat_long_concelho_data()
    # building a list (in the same order as the concelhos_format index) with the colors in which to fill the concelhos in the map
    # in this case we only fill the concelhos in the concelhos_mapped_together with the correspodent color of their BMU given by the neuron_coords given as argument
    # all they other concelhos will be attributed the color white, to just display their borders
    color_BMU = list(colors_per_neuron[neuron_coords])
    color_concelho = [
        color_BMU
        if (concelho in concelhos_mapped_together[0]) or concelho == concelho_selected
        else [1, 1, 1]
        for concelho in concelhos_format.index
    ]
    # adding the correspodent color for each concelho as a new column, respecting their BMU color unit
    concelhos_format["color_concelhos"] = color_concelho
    # plotting section
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(10, 15))
    # First, Plotting Portugal and its concelhos borders, and each concelho with the corresponding color of their BMU
    concelhos_format.plot(
        ax=ax, edgecolor="black", color=concelhos_format["color_concelhos"]
    )
    # setting up the image title, axis names, ticks etc
    ax.set_title(
        f"Concelhos Mapped in the Same Neuron",
        fontsize=20,
        fontweight="bold",
        y=1,
        loc="center",
    )
    ax.set_xlabel("Longitude", fontsize=15, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=15, labelpad=5)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.legend(bbox_to_anchor=(1.50, 1.01),loc = "upper right",fontsize=8)
    # plt.savefig(f'SOM Clustering Map of Concelhos in {altura_do_ano} Filled',dpi=100, bbox_inches = 'tight')
    # plt.show()
    return fig

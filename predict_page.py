# Importe de Todas as Bibliotecas Necessárias
import streamlit as st

from functions_explore_page import load_obj
from functions_predict_page import (
    classify,
    forecast_new_data_input_mappings,
    plot_concelhos_classified_together,
)


def load_predict_page(all_data):
    # create a title app
    st.image(r"Images/covid_19_logo.png", width=200)
    st.title("Covid-19 SOM Clustering Municipalities Predictions")
    # add some text to the app taking advantage of markup language

    st.write(
        """
    ### Want to Know the SOM Clustering Predictions for your Municipality?
    """
    )
    # create a selectorbox as a sidebar for the user in the app to use, to choose one of the 5 time intervals defined
    concelho = st.selectbox(
        "Select Municipality to Predict Mappings",
        tuple(all_data.index),
    )

    # computes a df where is shown for each 'concelho'(rows) the neurons where it was mapped to
    # at each of the time stages(columns)
    _, col2, _ = st.columns(3)
    ok = col2.button(f"Predict SOM Neuron Mappings of New Municipality")
    # if the user clicks, ok turns truthy
    if ok:  # true is the user of the web app wants a prediction
        # use the classification functions previouly defined to make the classification of the concelho chosen by the user
        mappings_forecasted = forecast_new_data_input_mappings(all_data, concelho)
        # st.write(
        # f"The Neuron Mappings of the {concelho} along all the timeframes are: {st.dataframe(mappings_forecasted)}"
        # )
        st.dataframe(mappings_forecasted.astype(str))

    st.markdown(
        f"""
    ### *Your Municipality was Mapped to a High Risk Neuron ? \t  Which Concelhos were Mapped to the Same Neuron ?*
    """
    )
    st.markdown(
        "###### __In What Timeframe Do You Wish to Perform a Risk Evaluation of your Municipality Mappings ?__"
    )
    time_frame = st.selectbox(
        "Select Time Period",
        (
            "1st Emergency State",
            "Summer Season",
            "September-October of 2020",
            "2 Wave of COVID-19",
            "Christmas and Holiday Season",
        ),
    )
    neuron_coords, concelhos_mapped_together = classify(all_data, concelho, time_frame)
    col1, col2 = st.columns(2)
    colors_per_neuron = load_obj("dic_colors_per_neuron_cartographic")
    dic_neurons_above_average = load_obj("dic_neurons_above_average")
    col1.pyplot(
        plot_concelhos_classified_together(
            concelhos_mapped_together, colors_per_neuron, neuron_coords, concelho
        )
    )
    with col2:
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        if neuron_coords in dic_neurons_above_average[time_frame]:
            st.error("__Mapped to a High Risk Neuron__")
        else:
            st.success("__Not mapped to a High Risk Neuron__")

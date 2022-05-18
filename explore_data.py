import streamlit as st

from functions_explore_page import (
    SOM_clustering_grid,
    SOM_clustering_map,
    load_obj,
    plot_concelhos_in_risk_neurons,
    plot_feature_distribution_in_Portugal_cmap,
    plot_raw_incidências_per_neuron_fill_between,
    processing_all_needed_data,
    processing_incidence_needed_data,
)


def show_explore_data_page(
    raw_data_Covid19, Data_incidences, all_data, geo_other_data_concelhos
):
    # create a title app
    st.image(r"Images/covid_19_logo.png", width=200)
    st.title("COVID-19 Geo-Spatial SOM Clustering Analysis in Portugal's Mainland")
    # add some text to the app taking advantage of markup language
    st.sidebar.markdown("____")
    st.sidebar.markdown(
        """
    ## __*Want to Explore the COVID-19 Spreading Geo-Spatially in which Time Period?*__
    """
    )
    # create a selectorbox as a sidebar for the user in the app to use, to choose one of the 5 time intervals defined
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
    # importing all the necessary data
    all_needed_data = processing_all_needed_data(all_data, time_frame)
    colors_per_neuron, dic_neurons_heat_map, dic_neurons_above_average = (
        load_obj("dic_colors_per_neuron_cartographic"),
        load_obj("dic_neurons_heat_map"),
        load_obj("dic_neurons_above_average"),
    )
    Data_incidences = processing_incidence_needed_data(Data_incidences, time_frame)
    # the plots are beggining to be generated and display
    # first will be display the som clustering grid side by side with the equivalent cartographic map
    st.write("\n")
    col1, col2 = st.columns((2.5, 2.5))
    with col1:
        # just to make the first grid be in the same horizontal line as the second figure
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.pyplot(SOM_clustering_grid(all_needed_data, time_frame, colors_per_neuron))
    with col2:
        st.pyplot(SOM_clustering_map(all_needed_data, time_frame, colors_per_neuron))
    # now we will plot the scatter plot of incidences per neuron
    st.write("\n")
    st.pyplot(
        plot_raw_incidências_per_neuron_fill_between(
            raw_data_Covid19, all_data, time_frame, colors_per_neuron
        )
    )
    st.write("")
    # cartographic map of the most affected geographical areas in this time period if user wants to and also the cartographic map of each of the additional
    # socio economic and demographic features
    col3, col4 = st.columns((2.5, 2.5))
    with col3:
        st.markdown("____")
        # just to make the first grid be in the same horizontal line as the second figure
        st.markdown(
            """### __*What Were The Geographical Areas Most Affected By COVID-19?*__
    """
        )
        st.markdown("____")
        st.write("\n")
        st.write("\n")
        # user_wants_to_plot_concelhos_in_risk_neurons = col3.button("Display High Risk Geographic Areas in this Time Period")
        # if the user clicks, ok turns truthy
        # if user_wants_to_plot_concelhos_in_risk_neurons:
        st.pyplot(
            plot_concelhos_in_risk_neurons(
                all_needed_data,
                time_frame,
                dic_neurons_above_average,
                colors_per_neuron,
            )
        )
    with col4:
        st.write("\n")
        st.write("\n")
        feature_to_plot = st.selectbox(
            "Select the Additional Socio-Economic and Demographic Feature You Are Most Interested In",
            (
                "Population Density",
                "Deprivation Score",
                "Youth Population",
                "Eldery Population",
                "Jobs in the Primary Sector",
                "Jobs in the Secondary Sector",
                "Jobs in the Tertiary Sector",
                "People Needing State Benefits",
                "Number of Schools",
            ),
        )
        user_wants_additional_features_geographic_distribution = col4.button(
            "Generate Feature Geographic Distribution in Portugal's Mainland"
        )
        # if the user clicks, ok turns truthy
        if user_wants_additional_features_geographic_distribution:
            st.pyplot(
                plot_feature_distribution_in_Portugal_cmap(
                    feature_to_plot, geo_other_data_concelhos
                )
            )

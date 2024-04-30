import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

### File upload ###
st.subheader("File upload")

file = st.file_uploader("Upload file here")

### Simple data visualization ###
if file:
    df = pd.read_csv(file)
    st.subheader("Data Exploration")
    if st.toggle("Show data"):
        st.write(df.shape)
        st.dataframe(df)
    if st.checkbox("Columns data types"):
        st.write(df.dtypes)
    if st.checkbox("Null values count"):
        st.write(df.isnull().sum().sort_values(ascending=False))

    #### Unnecessary columns removal ####
    cols = st.multiselect("Remove columns", df.columns)
    df.drop(columns=cols, axis=1, inplace=True)
    st.write("Columns", df.columns)

    #### Hadling missing values ####

    st.subheader("Handling missing values")

    # Detecting numerical columns
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    # Detecting categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.to_list()

    # Handling numerical missing values
    radio_num = st.radio(
        "Handle numerical missing values",
        ["Remove them", "Fill with a specific value", "Impute them"],
    )

    if radio_num == "Remove them":
        df[num_cols].dropna()
        st.write("Data new shape:", df.shape)
    elif radio_num == "Fill with a specific value":
        val = st.text_input("Value")
        df[num_cols].fillna(val, inplace=True)
    elif radio_num == "Impute them":
        df[num_cols] = df[num_cols].interpolate(
            method="linear", limit_direction="forward"
        )

    # Handling categorical missing values
    radio_cat = st.radio(
        "Handle Categorical missing values",
        ["Remove them", "Fill with a specific value"],
    )

    if radio_cat == "Remove them":
        df[cat_cols].dropna()
        st.write("Data new shape:", df.shape)
    elif radio_cat == "Fill with a specific value":
        val = st.text_input("Value")
        df[cat_cols].fillna(val, inplace=True)

    ### Choosing the target column ###
    st.subheader("Choose y-column")
    y = st.selectbox("Select y", df.columns)

    ### Categorical data Encoding ###
    st.subheader("Categorical data encoding")

    enc = st.radio("Select encoder", ["Label encoder", "One-Hot encoder"])

    if enc == "One-Hot encoder":
        df = pd.get_dummies(df, columns=cat_cols)
        st.write(df.shape)
        st.dataframe(df.sample(10))

    if enc == "Label encoder":
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        st.dataframe(df.sample(10))

    ### Creating the model ###

    st.subheader("ML model")
    # Detect classificatoion or regression
    if df[y].nunique() == 2:
        type = "class"
    else:
        type = "reg"

    # Creating the model using Pycaret

    # Classification model
    if type == "class":
        # st.write("class")
        from pycaret.classification import *

        exp = setup(df, target=y, categorical_features=cat_cols)
        best_model = compare_models()
        metrics = exp.pull()
        st.write("Best model Ranking")
        st.write(metrics)

        # Model prediction
        sample = predict_model(best_model, df.sample(10))
        st.write("Sample prediction", sample)

        # creating model file
        metrics.Model.tolist()
        model_name = str(metrics[:1]["Model"][0])

        # Saving best model
        # st.write("Save best model", save_model(best_model, model_name=model_name))

        # st.link_button(
        #     label="Download model", url=save_model(best_model, model_name=model_name)
        # )

        # ### Download best model ###
        # st.download_button(
        #     label="Download model",
        #     data=save_model(best_model, model_name=model_name),
        #     file_name="model",
        # )

    # Regression model
    elif type == "reg":
        # st.write("reg")
        from pycaret.regression import *

        exp = setup(df, target=y, categorical_features=cat_cols)
        best_model = compare_models()
        metrics = exp.pull()
        st.write("Best model Ranking")
        st.write(metrics)

        # Model prediction
        sample = predict_model(best_model, df.sample(10))
        st.write("Sample prediction", sample)

        # creating model file
        metrics.Model.tolist()
        model_name = str(metrics[:1]["Model"][0])

        # Saving best model
        # st.write("Save best model", save_model(best_model, model_name=model_name))

        # st.link_button(
        #     label="Download model", url=save_model(best_model, model_name=model_name)
        # )

        # ### Download best model ###
        # st.download_button(
        #     label="Download model",
        #     data=save_model(best_model, model_name=model_name),
        #     file_name="model",
        # )

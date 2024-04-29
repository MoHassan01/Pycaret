import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.subheader("File upload")

file = st.file_uploader("Upload file here")

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

    st.subheader("Handling missing values")

    cols = st.multiselect("Remove columns", df.columns)
    df.drop(columns=cols, axis=1, inplace=True)
    st.write("Columns", df.columns)

    radio = st.radio(
        "Handle missing values",
        ["Remove them", "Fill with a specific value", "Impute them"],
    )

    # Detecting numerical columns
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    if radio == "Remove them":
        df.dropna()
        st.write("Data new shape:", df.shape)
    elif radio == "Fill with a specific value":
        val = st.text_input("Value")
        df[num_cols].fillna(val, inplace=True)
    elif radio == "Impute them":
        df[num_cols] = df[num_cols].interpolate(
            method="linear", limit_direction="forward"
        )

    st.subheader("Choose y-column")
    y = st.selectbox("Select y", df.columns)

    # Data Encoding
    st.subheader("Categorical data encoding")

    # Detecting categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.to_list()

    enc = st.radio("Select encoder", ["Label encoder", "One-Hot encoder"])

    if enc == "One-Hot encoder":
        df = pd.get_dummies(df, columns=cat_cols)
        st.write(df.shape)
        st.dataframe(df.sample(10))

    if enc == "Label encoder":
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        st.dataframe(df.sample(10))

    # Detect classificatoion or regression
    if df[y].nunique() == 2:
        type = "class"
    else:
        type = "reg"

    st.write(type)

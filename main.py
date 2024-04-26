import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

    if radio == "Remove them":
        df.dropna()
        st.write("Data new shape:", df.shape)
    if radio == "Fill with a specific value":
        val = st.text_input("Value")
        df.fillna(val, inplace=True)

    st.subheader("Choose y-column")
    y = st.selectbox("Select y", df.columns)

    # Data Encoding
    st.subheader("Categorical data encoding")
    # Detecting categorical columns
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].to_numpy().reshape(-1, 1)
            cat_cols.append(col)

    enc = st.radio("Select encoder", ["Label encoder", "One-Hot encoder"])

    if enc == "Label encoder":
        enc = LabelEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols])

    if enc == "One-Hot encoder":
        enc = OneHotEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols])

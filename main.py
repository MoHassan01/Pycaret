import streamlit as st
import pandas as pd

st.text("Hellooooooooo")


# def load_data(data) -> pd.DataFrame:
#     return pd.read_csv(data)

file_upload = st.file_uploader("upload CSV")

if file_upload:
    st.write(pd.read_csv(file_upload))

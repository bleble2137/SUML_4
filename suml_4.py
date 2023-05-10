import pickle
import pathlib
import streamlit as st
from datetime import datetime

startTime = datetime.now()
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = 'model.sv'
model = pickle.load(open(filename, 'rb'))

sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}


def main():
    st.set_page_config(page_title="Serduszko nie puka w rytmie czaczy. S20240")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://i.kym-cdn.com/entries/icons/original/000/011/699/sad-broken-heart-l.png")

    with overview:
        st.title("Titanic")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2,
                                  format_func=lambda x: embarked_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])

    with right:
        age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=10)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=10)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=480, step=1)

    data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()

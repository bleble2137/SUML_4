import streamlit as st
import pickle

filename = 'model_.sv'
model = pickle.load(open(filename, 'rb'))


def main():
    st.set_page_config(page_title="Serduszko nie puka w rytmie czaczy. S20240")
    overview = st.container()
    right = st.container()
    prediction = st.container()


    st.image("https://i.kym-cdn.com/entries/icons/original/000/011/699/sad-broken-heart-l.png")
    
    with overview:
        st.title("Sercochoróbska predykcyja. S20240")
    
    with right:
        var1 = st.slider("Wiek:", value=40, min_value=28, max_value=100)
        var2 = st.slider("RBP", value=132, min_value=0, max_value=200)
        var3 = st.slider("Cholesterol", value=200, min_value=60, max_value=610)
        var4 = st.slider("Max Heart Rate", value=136, min_value=60, max_value=210)
        var5 = st.slider("Old peak", value=0.8, min_value=-3.0, max_value=7.0, step=0.1)
    
    data = [[var1, var2, var3, var4, var5]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)
    
    with prediction:
        st.subheader("Wynik")
        st.subheader(("Chory" if survival[0] == 1 else "Zdrowy"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()

import streamlit as st

st.title("Discussion")

st.header("")
st.header('Why would a predictive model be useful?')
st.write("Our model allows us to predict CPI changes and gain a greater understanding of what causes these changes. With so many Americans feeling the effects of inflation, predicting how long spikes will be and how severe they will be is crucial to understanding the United States’ economic outlook. Having this forecast also gives us an idea of what will happen to CPI dependent economic factors, such as wages, retirement benefits, and tax brackets.")

st.header("")
st.header("What difficulties did we face when training our model?")
st.write("The biggest weakness of our model is that we cannot factor in major socio-political events. The pandemic lockdowns cause erratic changes in CPI, and high levels of inflation. External factors such as the Russia-Ukraine conflict also introduce even more entropy to our forecasting. ")

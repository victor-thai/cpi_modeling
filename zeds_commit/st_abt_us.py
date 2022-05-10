from PIL import Image
import streamlit as st

st.title("About Us")

banner = Image.open("ds3_banner.PNG")
st.image(banner, width = 700)

st.write("We are a group of students from UCSD's Data Science Student Society. As members of the CPI project team, we show passion, initiative, and competence to contribute to this 2-quarter long project. Our team has a diverse background but we are brought together by a common love for data science.")

st.title("Project Members")

v_col1, v_col2 = st.columns(2)
with v_col1:
    st.subheader("Project Lead: Victor Thai")
    st.write("Victor is a second-year at UC San Diego, majoring in Data Science and minor in Cognitive Science. He is a member of CASA at UCSD and enjoys weightlifting to relieve stress.")
with v_col2:
    st.subheader("")
    v_img = Image.open("victor_headshot.jpeg")
    st.image(v_img, width = 200)

m_col1, m_col2 = st.columns(2)
with m_col1:
    st.subheader("Michael Chen")    
    st.write("Michael is a second-year student at UC San Diego, majoring in Data Science and Minor in Management Science. Michael was in the Chinese Union. A fun fact about him is that he gets excited when facing a hard task to be solved.")
with m_col2:
    st.subheader("")
    m_img = Image.open("michael_headshot.jpeg")
    st.image(m_img, width = 200)
    
s_col1, s_col2 = st.columns(2)
with s_col1:
    st.subheader("Stephanie Chavez")
    st.write("Stephanie is a second-year student at UC San Diego, majoring in Data Science. Stephanie is a sister of Sigma Pi Alpha Sorority, Inc. and is a First-Gen Student Success Peer Coach. Some of her interests outside of data science include musical theatre, fantasy books, and dogs.")


z_col1, z_col2 = st.columns(2)
with z_col1:
    st.subheader("Zed Siyed")
    st.write("Zed is a first-year student at UC San Diego, majoring in Computer Science. Zed is also a member of Triton Consulting Group as the VP Tech Consulting. Some of Zed's hobbies are sports analytics and weightlifting.")
with z_col2:
    st.subheader("")
    zed_img = Image.open("zed_headshot.PNG")
    st.image(zed_img, width = 200)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import multipage_template_streamlit as multipage

#wide layout
st.set_page_config(layout='wide')

#Clears the cache when the app is started
multipage.start_app()

#multipage object
app = multipage.MultiPage()
app.start_button = "Let's explore this!"
app.navbar_name = "Table of Contents"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"


# intro page function to be called for page
def intropage():
	st.header('CPI Modeling')

	st.markdown("# What will our economy look like in the next month?")

	intro_left_col, intro_right_col = st.columns(2)

	with intro_right_col:
		st.write("This project focues on analyzing the trends of Consumer \
				Price Index, also known as CPI, and applies a predictive VAR model \
				 to predict the next month's CPI.")

	with intro_left_col:
		st.image('cpi_streamlit_photos/cpi_intro_photo.jpeg')



# about us page
def aboutuspage(prev_page):
	st.title("About Us")

	banner = Image.open("cpi_streamlit_photos/ds3_banner.PNG")
	st.image(banner, width = 700)

	st.write("We are a group of students from UCSD's Data Science Student Society. As members of the CPI project team, we show passion, initiative, and competence to contribute to this 2-quarter long project. Our team has a diverse background but we are brought together by a common love for data science.")

	st.title("Project Members")

	v_col1, v_col2 = st.columns(2)
	with v_col1:
	    st.subheader("Project Lead: Victor Thai")
	    st.write("Victor is a second-year at UC San Diego, majoring in Data Science and minor in Cognitive Science. He is a member of CASA at UCSD and enjoys weightlifting to relieve stress.")
	with v_col2:
	    st.subheader("")
	    v_img = Image.open("cpi_streamlit_photos/victor_headshot.jpeg")
	    st.image(v_img, width = 200)

	m_col1, m_col2 = st.columns(2)
	with m_col1:
	    st.subheader("Michael Chen")    
	    st.write("Michael is a second-year student at UC San Diego, majoring in Data Science and Minor in Management Science. Michael was in the Chinese Union. A fun fact about him is that he gets excited when facing a hard task to be solved.")
	with m_col2:
	    st.subheader("")
	    m_img = Image.open("cpi_streamlit_photos/michael_headshot.jpeg")
	    st.image(m_img, width = 200)
	    
	s_col1, s_col2 = st.columns(2)
	with s_col1:
	    st.subheader("Stephanie Chavez")
	    st.write("Stephanie is a second-year student at UC San Diego, majoring in Data Science. Stephanie is a sister of Sigma Pi Alpha Sorority, Inc. and is a First-Gen Student Success Peer Coach. Some of her interests outside of data science include musical theatre, fantasy books, and dogs.")
	with s_col2:
		st.subheader("")
		steph_img = Image.open("cpi_streamlit_photos/steph_headshot.jpg")
		st.image(steph_img, width = 200)

	z_col1, z_col2 = st.columns(2)
	with z_col1:
	    st.subheader("Zed Siyed")
	    st.write("Zed is a first-year student at UC San Diego, majoring in Computer Science. Zed is also a member of Triton Consulting Group as the VP Tech Consulting. Some of Zed's hobbies are sports analytics and weightlifting.")
	with z_col2:
	    st.subheader("")
	    zed_img = Image.open("cpi_streamlit_photos/zed_headshot.PNG")
	    st.image(zed_img, width = 200)


app.set_initial_page(intropage)
# app.add_app("Home page", intropage)
app.add_app("About Us", aboutuspage)

app.run()
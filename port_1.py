import streamlit as st
import base64
import pandas as pd
import numpy as np
from project_page import projects
from certificates_page import certificates

# App title
st.set_page_config(page_title="Data Science Portfolio", page_icon="📊")

# --- HEADER SECTION ---

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Image path
image_path = r"images\profile pic.jpg"  # Use forward slashes or raw strings to avoid path issues

# Get the base64 encoded image
encoded_image = get_base64_image(image_path)

# Display the image in the sidebar
st.sidebar.markdown(f"""
    <style>
    .sidebar-profile-pic {{
        top: 20px;
        right: 10px;
        width: 100%;
        height: auto;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #fff;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }}
    .sidebar-name {{
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 15px;
    }}
    </style>
    <img src="data:image/jpg;base64,{encoded_image}" class="sidebar-profile-pic">
    <div class="sidebar-name">Anandakrishnan N</div>
    """, unsafe_allow_html=True)

page = st.sidebar.radio("Explore",["About me", "Skills & Education", "Projects", "Certificates"])

if page == "About me":
    st.title("Hi, I'm Anandakrishnan!")

    # Introduction
    st.markdown("""
    I am a **Data Scientist** with expertise in turning raw data into actionable insights using various Machine Learning models and analytical techniques.
    With a background in Python programming and data analytics, I have worked on numerous projects that involve **data processing**, **visualization**, and **predictive modeling**.

    ### My Mission
    To explore complex data sets and deliver practical solutions that can drive meaningful impact in business decision-making.

    ### What I Love to Do
    - **Data Analysis**: Digging deep into datasets to discover hidden trends and patterns.
    - **Machine Learning**: Creating smart models that learn and improve over time.
    - **Visualization**: Presenting data in a way that tells a compelling story.

    Feel free to explore the rest of my portfolio!
    """)
    # Resume Download Section
    st.header("My Resume")

    resume_file_path = "files\Anand resume 20 sep.pdf"

    # Function to read the PDF file
    def read_pdf(file_path):
        with open(file_path, "rb") as pdf_file:
            return pdf_file.read()

    # Read the resume file
    resume_data = read_pdf(resume_file_path)

    # Display the download button for the resume
    st.download_button(
        label="Download Resume",
        data=resume_data,
        file_name="Anandakrishnan_Resume.pdf",  # Set the name for the downloaded file
        mime="application/pdf"
    )

    # Footer
    st.markdown("---")
    st.markdown("Find me on [LinkedIn](https://www.linkedin.com/in/anandakrishnan-19bma114-4135852a5?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [GitHub](https://github.com/Anand-krish-0) | [Twitter](https://twitter.com/yourprofile)")

elif page == "Skills & Education":
    tab1, tab2 = st.tabs(["Skills", "Education"])

    # Skills Tab
    with tab1:
        st.header("Skills")
        skills = {
            "Programming Languages": "Python, SQL",
            "Libraries & Frameworks": "NumPy, Pandas, Scipy, Scikit-Learn, Streamlit",
            "Data Visualization": "Matplotlib, Seaborn, Power BI, Tableau",
            "Data Processing Tools": "Excel, SQL",
            "Others": "Version Control (GitHub), Jupyter Notebooks, Excel, PowerPoint"
        }

        for skill, details in skills.items():
            st.write(f"- **{skill}**: {details}")

    # Education Tab
    with tab2:
        st.header("Education")
        st.markdown("""
        - **Data Science and Python**, Qtree Technologies, Coimbatore, 2024
        - **Master of Science** in Statistics, Government Arts College, Coimbatore, 2024
        - **Bachelor of Science** in Mathematics, Government Arts College, Coimbatore, 2022
        """)

elif page == "Projects":
    projects()

elif page == "Certificates":
    certificates()

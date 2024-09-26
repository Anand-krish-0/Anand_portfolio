import streamlit as st
import base64
import os
import pandas as pd
import numpy as np
from project_page import projects
from certificates_page import certificates

# App title
st.set_page_config(page_title="Anand's Portfolio", page_icon="📊")

# CSS for professional background
st.markdown("""
    <style>
    body {
        background-color: #87CEEB;  /* Light sky blue background */
    }
    .sidebar .sidebar-content {
        background-color: #2C3E50;  /* Darker sidebar background */
        color: white;
    }
    .sidebar-profile-pic {
        top: 20px;
        right: 10px;
        width: 100%;
        height: auto;
        border-radius: 0;  /* Set to 0 for square shape */
        object-fit: cover;
        border: 2px solid #fff;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar-name {
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 15px;
        color: #ecf0f1;  /* Light color for text */
    }
    h1, h2, h3 {
        color: #34495e;  /* Darker color for headers */
    }
    p {
        color: #2c3e50;  /* Darker color for paragraph text */
    }
    </style>
""", unsafe_allow_html=True)


# --- HEADER SECTION ---

# Function to convert image to base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
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
        width: 100px;  /* Set a fixed width */
        height: 100px; /* Set a fixed height */
        border-radius: 15px;  /* Set to a value for rounded corners */
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

#I am a **Data Scientist** with expertise in turning raw data into actionable insights using various Machine Learning models and analytical techniques.
# With a background in Python programming and data analytics, I have worked on numerous projects that involve **data processing**, **visualization**, and **predictive modeling**.

    # Introduction
    st.markdown("""
                 I’m Anandakrishnan N, a dedicated Data Science fresher with a strong foundation 
                 in data analysis, machine learning, and statistical modeling. Proficient in Python, 
                 SQL, and popular machine learning libraries, I have hands-on experience building 
                 predictive models and uncovering insights from complex datasets.

                 With a keen interest in solving real-world problems through data, I am eager to 
                 contribute to a forward-thinking team and grow as a data professional. Looking forward 
                 to starting my career in data science, I aim to collaborate with teams that value 
                 innovation and use data to drive impactful decisions. Passionate about continuous 
                 learning, I’m excited to apply my skills in a dynamic environment.

                 ### My Mission
                 To explore complex data sets and deliver practical solutions that can drive meaningful impact in business decision-making.

                 ### What I Love to Do
                 - **Data Analysis**: Digging deep into datasets to discover hidden trends and patterns.
                 - **Machine Learning**: Creating smart models that learn and improve over time.
                 - **Visualization**: Presenting data in a way that tells a compelling story.

                Feel free to explore the rest of my portfolio!
                """)
    # Resume Download Section
    st.subheader("My Resume")

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
    # Display contact information
    st.subheader("To Contact:")

    # Display phone number
    st.write("📞 **Phone Number**: +91 7550356881")

    # Display email address
    st.write("✉️ **Email**: anandakrishnanips2001@gmail.com")

    # Optional: Add links to social media or website (if applicable)
    #st.write("""
    #For more details, you can visit my [LinkedIn Profile](https://www.linkedin.com/in/anandakrishnan-19bma114) or my [GitHub Profile](https://github.com/Anand-krish-0).
    #         """)
    # Updated GitHub and LinkedIn logos
    github_logo_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"  # Latest GitHub logo
    linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"  # LinkedIn logo

    # Display clickable GitHub logo
    st.markdown(f"""
        <p>For more details, you can visit my:</p>
        <a href="https://github.com/Anand-krish-0" target="_blank">
            <img src="{github_logo_url}" alt="GitHub" style="width:30px; height:30px;">
        </a> 
        <a href="https://github.com/Anand-krish-0" target="_blank">GitHub Profile</a>
        <br>
        <a href="https://www.linkedin.com/in/anandakrishnan-19bma114" target="_blank">
            <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:30px; height:30px;">
        </a> 
        <a href="https://www.linkedin.com/in/anandakrishnan-19bma114" target="_blank">LinkedIn Profile</a>
        """, unsafe_allow_html=True)

    #st.markdown("Find me on [GitHub](https://github.com/Anand-krish-0) | [LinkedIn](https://www.linkedin.com/in/anandakrishnan-19bma114) ")

elif page == "Skills & Education":
    tab1, tab2 = st.tabs(["Skills", "Education"])

    # Skills Tab
    with tab1:
        st.header("Skills")
        skills = {
            "Programming Languages": "Python, SQL",
            "Machine Learning": (
                "Supervised Learning: Linear Regression, Decision Trees, Random Forest, etc., "
                "Unsupervised Learning: K-means Clustering, Principal Component Analysis (PCA), "
                "Ensemble Methods: Bagging, Boosting, "
                "Model Evaluation: Cross-Validation, "
                "Feature Selection: Lasso Regression, Ridge Regression, "
                "Optimization: Gradient Descent, Hyperparameter Tuning"
            ),
            "Libraries & Frameworks": "NumPy, Pandas, Scipy, Scikit-Learn, Streamlit",
            "Data Visualization": "Matplotlib, Seaborn, Power BI, Tableau",
            "Data Processing Tools": "Excel, SQL Server, MySQL",
            "Others": "Version Control (GitHub), Jupyter Notebooks, Excel, PowerPoint"
        }

        # Display the skills in a formatted way
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

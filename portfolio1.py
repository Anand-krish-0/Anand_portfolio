import streamlit as st
import pandas as pd
import numpy as np
from project_page import projects
from streamlit_option_menu import option_menu  # Import option_menu

# App title
st.set_page_config(page_title="Data Science Portfolio", page_icon="📊")

# --- HEADER SECTION ---
st.title('Data Science Portfolio')
st.subheader("Hi! I'm Anandakrishnan.")
st.markdown("""
I'm passionate about uncovering insights from data and building machine learning models that make impactful decisions.
In this portfolio, you'll find some of the work I've done in the areas of data analysis, machine learning, and data visualization.
""")

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    pages = ["About me", "Skills", "Projects"]
    nav_tab = option_menu(
        menu_title="Anandakrishnan",
        options=pages,  # Corrected the options to be the list `pages`
        icons=['person-fill', 'file-text', 'briefcase'],  # Number of icons should match number of options
        menu_icon="cast", 
        default_index=0,
    )

# --- SKILLS SECTION ---
st.header("Skills")
skills = {
    "Programming Languages": "Python, SQL",
    "Libraries & Frameworks": "Pandas, NumPy, Scipy, Scikit-Learn",
    "Data Visualization": "Matplotlib, Seaborn, Power BI, Tableau",
    "Data Processing Tools": "Excel, SQL",
    "Others": "Version Control (Git), Streamlit, Jupyter Notebooks"
}

for skill, details in skills.items():
    st.write(f"- **{skill}**: {details}")

# --- PROJECTS SECTION ---
st.header("Projects")

# Example Project 1: Data Analysis
st.subheader("Project 1: E-commerce Customer Segmentation")
st.markdown("""
- **Description**: Analyzed e-commerce customer data to identify customer segments based on purchasing behavior.
- **Tools**: Python, Pandas, Scikit-learn, Matplotlib
- **Key Insights**: Identified 3 key customer segments using K-means clustering.
""")
st.image("project_images/ecommerce_segmentation.png", caption="Customer Segmentation Clustering Results")

# Example Project 2: Machine Learning
st.subheader("Project 2: House Price Prediction")
st.markdown("""
- **Description**: Built a machine learning model to predict house prices based on various features.
- **Tools**: Python, Scikit-learn, XGBoost, Matplotlib
- **Model Performance**: Achieved an R² score of 0.85 on the test set.
""")
st.image("project_images/house_price_prediction.png", caption="House Price Prediction Model")

# Example Project 3: Deep Learning
st.subheader("Project 3: Image Classification using CNNs")
st.markdown("""
- **Description**: Built a convolutional neural network to classify images from the CIFAR-10 dataset.
- **Tools**: Python, TensorFlow, Keras
- **Model Accuracy**: Achieved an accuracy of 90% on the test set.
""")
st.image("project_images/image_classification_cnn.png", caption="CNN Model Architecture")

# --- CONTACT SECTION ---
st.header("Contact")
st.write("Feel free to reach out if you want to collaborate on projects or discuss opportunities.")
st.write("📧 Email: [your_email@example.com](mailto:your_email@example.com)")
st.write("🌐 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)")

# Optional: Contact form
st.subheader("Get In Touch")
contact_form = """
<form action="https://formspree.io/f/{your_form_id}" method="POST">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="_replyto" placeholder="Your email" required>
    <textarea name="message" placeholder="Your message"></textarea>
    <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)

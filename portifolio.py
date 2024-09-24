import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "Projects", "Skills", "Contact"])

# 1. Home / Introduction Page
if page == "Home":
    st.title("Data Science Portfolio")
    st.write("""
    Hi! I am Anandakrishnan, a data scientist with a passion for data-driven solutions. 
    This portfolio showcases some of my favorite projects and the skills I've developed along the way. 
    Feel free to explore!
    """)

# 2. Projects Section
elif page == "Projects":
    st.title("Projects")
    
    # Example Project: Titanic Dataset Analysis
    st.header("1. Titanic Survival Prediction")
    st.write("""
    In this project, I analyzed the Titanic dataset to predict whether a passenger would survive 
    based on features such as age, gender, and class.
    """)
    
    # Load the Titanic dataset
    @st.cache
    def load_data():
        return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    
    df = load_data()
    
    # Show the dataset
    st.subheader("Dataset")
    st.write(df.head())
    
    # Display data summary
    st.subheader("Data Summary")
    st.write(df.describe())
    
    # Visualization: Survival rate by gender
    st.subheader("Survival Rate by Gender")
    fig, ax = plt.subplots()
    sns.barplot(x="Sex", y="Survived", data=df, ax=ax)
    st.pyplot(fig)

    # Other project examples (you can add more similarly)
    
# 3. Skills Section
elif page == "Skills":
    st.title("Skills")
    st.write("""
    - **Programming Languages**: Python, SQL
    - **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
    - **Data Visualization**: Seaborn, Plotly, Matplotlib
    - **Machine Learning**: Regression, Classification, Clustering
    - **Tools**: Jupyter, Git, Streamlit
    """)

# 4. Contact Section
elif page == "Contact":
    st.title("Contact Me")
    
    # Simple contact form
    st.write("Feel free to reach out via email or LinkedIn!")
    
    # Inputs for contact
    with st.form(key='contact_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.form_submit_button(label="Send")
        
    if submit_button:
        st.success(f"Thank you {name}, your message has been sent!")


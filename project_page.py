import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
from PIL import Image

def projects():
    st.title("Project")

    # Create tabs correctly
    tab1, tab2, tab3, tab4 = st.tabs(["Awareness of EV's in Tamil Nadu", "Streamlit Projects",
                                      "ML projects", "Data Analysis projects"])

    with tab1:
        st.header("Awareness level of Electric Vehicles in Tamilnadu")
        st.write("""
            In this project, I analyzed the awareness level of Electric Vehicles in Tamil Nadu. 
            Data are collected directly from people using a Google form.
        """)

        # Load dataset
        @st.cache_data
        def load_data():
            return pd.read_csv("datasets/responces.csv")
        
        df = load_data()

        # Create a labeled selectbox
        opts = st.selectbox("Select an analysis option", ["About dataset", "Univariate analysis", "Bivariate Analysis"])

        if opts == "About dataset":
            # Show the dataset
            st.subheader("1. Dataset")
            st.write(df.head())
            
            # Display data summary
            st.subheader("2. Data Summary")
            st.write(df.describe(include='all'))

        elif opts == "Univariate analysis":
            st.header("Univariate Analysis")

            # Visualization: Frequency of Gender
            st.subheader("1. Frequency of Respondent's Gender")
            fig, ax = plt.subplots()
            sns.histplot(df['Gender'], kde=False, bins=5)
            plt.xlabel('Gender')
            plt.ylabel('Frequency')
            st.pyplot(fig)

            # Visualization: Education Qualification
            st.subheader("2. Educational Qualification of Respondents")
            edu_q = df['Education Qualification '].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(edu_q, labels=edu_q.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

            # Visualization: Types of vehicles
            st.subheader("3. Types of Vehicles currently Using by Respondents")
            type_of_vehicles = df.iloc[:, 7:8].value_counts()
            fig, ax = plt.subplots()
            ax.pie(type_of_vehicles, labels=type_of_vehicles.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

        elif opts == "Bivariate Analysis":
            st.header("Bivariate Analysis")

            # Load and display images with corrected file paths (use forward slashes or raw strings)
            st.subheader("1. Types of vehicles used groubed by area")
            image1_url = r"images\Types of vehicles used groubed by area.png"
            st.image(image1_url, caption="Types of vehicles used grouped by area", use_column_width=True)

            st.subheader("2. Types of vehicle used grouped by educational qualification")
            image3_url = r"images\Types of vehicle used grouped by educational qualification.png"
            st.image(image3_url, caption="Types of vehicles used grouped by education", use_column_width=True)

            st.subheader("3. Types of vehicle grouped by employment status")
            image4_url = r"images\Types of vehicle grouped by employment status.png"
            st.image(image4_url, caption="Types of vehicles grouped by employment status", use_column_width=True)

    with tab2:


        st.title("Predictive Systems Created Using Streamlit")

        with st.container():
            # 1. Diabetes Prediction
            st.subheader("1. Diabetes Prediction")
            st.write("""The objective of this project is to build a support vector machine model that can predict the diabetes based on the given features. 
                     This model can be useful for finding the patients have diabetes or not. 
                     The features includes pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetiesPedigreeFunction and Age.
                     """)
            st.markdown("Model performance:")
            st.success("""The support vector machine model achieved an accuracy of 77.27% in prediction.""")
            def open_diabetes_website():
                webbrowser.open_new_tab("https://anand-krish-0-diabetics-app-seyp9h.streamlit.app/")
            if st.button("Open Diabetes Prediction App"):
                open_diabetes_website()

        with st.container():

            # 2. Loan Status Prediction
            st.subheader("2. Loan Status Prediction")
            st.write("""This project aims to predict whether a loan applicant will be approved for a loan based on 
                     several input features such as Gender, Martial status, Dependents, Self_employed, ApplicantIncome, 
                     CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit histor and Property Area. 
                     The model used is a Support Vector Machine (SVM), which is a supervised machine learning algorithm""")
            st.markdown("Model performance:")
            st.success("""The support vector machine model achieved an accuracy of 83.33% in prediction.""")
            
            def open_loan_website():
                webbrowser.open_new_tab("https://anand-krish-0-2-loan-status-prediction-ml5-predic-page-hjqcwk.streamlit.app/")
            if st.button("Open Loan Status Prediction App"):
                open_loan_website()

        with st.container():
            # 3. House Price Prediction
            st.subheader("3. House Price Prediction")
            
            st.write("""This project aims to predict house prices based on various features such as the size of 
                     the house, number of bedrooms, location, age of the property, and other relevant factors. 
                     The model used is the XGBoost Regressor, a powerful and efficient gradient boosting 
                     algorithm.""")
            st.write(""" 
                     XGBoost works by building an ensemble of decision trees, where each new tree 
                     corrects the errors of the previous one. It is particularly effective for 
                     regression tasks and handles both structured and unstructured data well. 
                     By learning from historical house data, the model predicts the continuous target 
                     variable—house price—with high accuracy.

                     This model is highly optimized for speed and performance, making it suitable for real-world 
                     applications in the real estate market, where accurate house price predictions can support 
                     decision-making for buyers, sellers, and investors.

                    """)
            def open_house_website():
                webbrowser.open_new_tab("https://anand-krish-0-3-house-price-predi-house-price-prediction-e5gi5d.streamlit.app/")
            if st.button("Open House Price Prediction App"):
                open_house_website()
        
        with st.container():
            # 4. Fake News Prediction
            st.subheader("4. Fake News Prediction")
            st.write("""
                     This project focuses on detecting fake news by analyzing the text of news articles. 
                     The model leverages NLTK (Natural Language Toolkit) for text preprocessing, TfidfVectorizer to 
                     convert the text into numerical features based on term frequency-inverse document frequency 
                     (TF-IDF), and Logistic Regression for classification.

                     By training on labeled datasets, the model learns to differentiate between real and fake 
                     news, making predictions based on the textual content, aiding in identifying misleading or 
                     false information in media
                     """)
            st.markdown("Model preperation:")
            st.success("The Logistic Regression model achieved an accuracy of 97.93% in prediction.")
            def open_fake_news_website():
                webbrowser.open_new_tab("https://anand-krish-0-4-fake-news-predictio-fake-news-prediction-0v5nel.streamlit.app/")
            if st.button("Open Fake News Prediction App"):
                open_fake_news_website()
        
        with st.container():
            # 5. Car Price Prediction
            st.subheader("5. Car Price Prediction")
            st.write("""
                     This project aims to predict the selling price of used cars using 
                     Lasso Regression, a linear model that applies L1 regularization to prevent 
                     overfitting and handle feature selection. The model analyzes various features 
                     like the car's age, mileage, engine size, fuel type, and more, to make accurate 
                     predictions.

                     By penalizing less important features, Lasso Regression simplifies the model, 
                     ensuring robust predictions while avoiding unnecessary complexity, making it 
                     an ideal choice for estimating car prices in dynamic markets.
                     """)
            if st.button("Open Car Price Prediction App"):
                st.markdown('<a href="https://anand-krish-0-5-car-price-predictio-car-price-prediction-v3uyxy.streamlit.app/" target="_blank">Click here</a>', unsafe_allow_html=True)

        with st.container():
            # 6. Wine Quality Prediction
            st.subheader("6. Wine Quality Prediction")
            st.write("""
                     In this project, a Random Forest Classifier is employed to predict the quality of 
                     wine based on various chemical properties like acidity, sugar content, pH levels, 
                     and alcohol concentration. The model creates multiple decision trees and combines 
                     their outputs to enhance prediction accuracy.

                     By leveraging the ensemble technique, the Random Forest algorithm provides 
                     robust and reliable predictions, making it effective for determining wine 
                     quality with high precision.
                     """)
            st.markdown("Model performance")
            st.success("The Random Forest Classifier model achieved an accuracy of 92.81% in prediction.")
            if st.button("Open Wine Quality Prediction App"):
                webbrowser.open_new_tab("https://anand-krish-0-5-wine-quality-pre-wine-quality-prediction-glroej.streamlit.app/")

        with st.container():
            # 7. Gold Price Prediction
            st.subheader("7. Gold Price Prediction")
            st.write("""
                     This project utilizes a Random Forest Regressor to predict gold prices based on 
                     historical market data and other financial indicators. The model constructs 
                     multiple decision trees and averages their predictions to deliver accurate and 
                     stable forecasts of gold prices.

                     By capturing complex patterns in the data, the Random Forest algorithm ensures 
                     reliable and robust price predictions, aiding in better investment decisions.
                     """)
            st.warning("Model: Random Forest Regressor")
            if st.button("Open Gold Price Prediction App"):
                webbrowser.open_new_tab("https://anand-krish-0-7-gold-price-predict-gold-price-prediction-dxpte7.streamlit.app/")

        with st.container():
            # 8. Heart Disease Prediction
            st.subheader("8. Heart Disease Prediction")
            st.write("""
                     This project uses a Logistic Regression model to predict the likelihood of heart 
                     disease in individuals based on key health indicators such as age, cholesterol 
                     levels, blood pressure, and other medical features. The model provides a binary 
                     output, helping identify patients at risk of heart disease with high accuracy.

                     Logistic regression is well-suited for this task due to its efficiency in binary 
                     classification problems, making it a reliable tool for medical predictions.
                     """)
            st.markdown("Model performance:")
            st.success("The Logistic Regression model achieved an accuracy of 90.16% in prediction.")
            if st.button("Open Heart Disease Prediction App"):
                webbrowser.open_new_tab("https://anand-krish-0-8-heart-disease-p-heart-disease-prediction-xxsppv.streamlit.app/")
        with st.container():
            # 9. Medical Insurance Cost Prediction
            st.subheader("9. Medical Insurance Cost Prediction")
            st.write("""
                     This project uses Linear Regression to predict medical insurance costs based on 
                     factors such as age, sex, body mass index (BMI), number of children, smoking status,
                      and region. The model establishes a linear relationship between these features and 
                     the insurance premium, enabling accurate cost estimates.

                     By analyzing historical data, the Linear Regression model helps identify key drivers 
                     of insurance costs, providing insights for individuals and insurers alike.
                     """)
            if st.button("Open Medical Insurance Cost Prediction App"):
                webbrowser.open_new_tab("https://anand-krish-0-9-medica-medical-insurance-cost-prediction-69bjvr.streamlit.app/")
        
        with st.container():
            # 10. Spam Mail Prediction
            st.subheader("10. Spam Mail Prediction")
            st.write("""
                     This project utilizes TfidfVectorizer to convert email text into numerical features and 
                     Logistic Regression to classify emails as spam or not. By analyzing the frequency and 
                     importance of words in the email content, the model effectively distinguishes between 
                     legitimate and spam emails, improving email filtering accuracy.
                    """)
            st.markdown("Model performance:")
            st.success("The Logistic Regression model achieved an accuracy of 96.68% in prediction.")
            if st.button("Open Spam Mail Prediction App"):
                webbrowser.open_new_tab("https://anand-krish-0-spam-mail-prediction-spam-mail-prediction-tusyy0.streamlit.app/")
    
        if st.button("View Raw Files"):
            webbrowser.open_new_tab("https://github.com/Anand-krish-0/q_l_project/tree/main/ML_project_streamlit")
    #ML projects
    with tab3:
        st.title("Machine Learning Projects")
        st.write("""
                    Machine learning projects focus on developing models that enable systems to learn from data and make 
                 informed predictions or decisions without explicit programming. These projects typically involve 
                 data preprocessing, feature engineering, and the use of algorithms such as classification, 
                 regression, or clustering. Common techniques include support vector machines, decision trees, 
                 linear and logistic regression, random forests, and more. Machine learning projects are applied in 
                 various fields, including healthcare, finance, marketing, and natural language processing, solving 
                 problems like predictive analytics, classification, and recommendation systems.
                 """)
        if st.button("View Machine Learning Projects"):
            webbrowser.open_new_tab("https://github.com/Anand-krish-0/q_l_project/tree/main/machine_learning")
    with tab4:
        st.title("Data Analysis Projects")
        st.write("""
                 Data analysis projects involve collecting, processing, and interpreting data to uncover insights, 
                 patterns, and trends that can inform decision-making. These projects typically include stages like 
                 data cleaning, exploratory data analysis (EDA), visualization, and statistical analysis. Tools like 
                 Python and visualization libraries such as Matplotlib, Seaborn are commonly used. 
                 The goal of data analysis is to identify key relationships, test hypotheses, and derive actionable 
                 insights from raw data across domains such as business, healthcare, finance, and social sciences. 
                 These projects often lead to more informed strategies and better outcomes
                 """)
        if st.button("View Data Analysis Projects"):
            webbrowser.open_new_tab("https://github.com/Anand-krish-0/EDA-lilst")
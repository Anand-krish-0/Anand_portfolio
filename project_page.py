import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import base64
from io import BytesIO
from PIL import Image

def projects():
    st.title("Projects")

    # Create tabs correctly
    tab1, tab2, tab3, tab4 = st.tabs(["Awareness of EV's in Tamil Nadu", "Streamlit Projects",
                                      "ML projects", "Data Analysis projects"])

    with tab1:
        st.header("Awareness level of Electric Vehicles in Tamilnadu")
        st.write("""
            In this project, I analyzed the awareness level of Electric Vehicles in Tamil Nadu. 
            Data are collected directly from people using a Google form.
        """)
        st.write("View raw project files through GitHub")

        if st.button("View Raw Files"):
                import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import base64
from io import BytesIO
from PIL import Image

def projects():
    st.title("Projects")

    # Create tabs correctly
    tab1, tab2, tab3, tab4 = st.tabs(["Awareness of EV's in Tamil Nadu", "Streamlit Projects",
                                      "ML projects", "Data Analysis projects"])

    with tab1:
        st.header("Awareness level of Electric Vehicles in Tamilnadu")
        st.write("""
            In this project, I analyzed the awareness level of Electric Vehicles in Tamil Nadu. 
            Data are collected directly from people using a Google form.
        """)
        st.write("View raw project files through GitHub")

        if st.button("View Raw Files"):
                st.markdown(
                    '<a href="https://github.com/Anand-krish-0/A-study-on-awareness-and-perceptionn-among-the-users-of-EVs" target="_blank">View Raw Files</a>',
                    unsafe_allow_html=True
                    )


        # Load dataset
        @st.cache_data
        def load_data():
            return pd.read_csv("datasets/responces.csv")
        
        df = load_data()

        # Create a labeled selectbox
        opts = st.selectbox("Select an analysis option", ["About dataset", "Univariate analysis", "Bivariate Analysis","Chi-square Test","Results", "Final Conclusion"])

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

            # Helper function to load and convert images to Base64
            def load_image(image_path):
                try:
                    image = Image.open(image_path)  # Open the image file
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")  # Convert image to bytes
                    img_str = base64.b64encode(buffered.getvalue()).decode()  # Encode as Base64
                    return img_str
                except Exception as e:
                    st.error(f"Error loading image {image_path}: {e}")
                    return None

            # Display images with Base64 encoding
            st.subheader("1. Types of vehicles used grouped by area")
            image1_path = "images/Types of vehicles used groubed by area.png"
            img_str1 = load_image(image1_path)
            if img_str1:
                st.image(f"data:image/png;base64,{img_str1}", caption="Types of vehicles used grouped by area", use_column_width=True)

            st.subheader("2. Types of vehicles used grouped by educational qualification")
            image3_path = "images/Types of vehicle used grouped by educational qualification.png"
            img_str3 = load_image(image3_path)
            if img_str3:
                st.image(f"data:image/png;base64,{img_str3}", caption="Types of vehicles used grouped by education", use_column_width=True)

            st.subheader("3. Types of vehicles grouped by employment status")
            image4_path = "images/Types of vehicle grouped by employment status.png"
            img_str4 = load_image(image4_path)
            if img_str4:
                st.image(f"data:image/png;base64,{img_str4}", caption="Types of vehicles grouped by employment status", use_column_width=True)
        #Chi-square Test
        elif opts == "Chi-square Test":
            st.title("Chi Square Test - Independence of Attributes")
            st.subheader("Objective of the Study:")
            st.write("""
                     To study about the awareness level of Electric Vehicles in the state of Tamil Nadu.
                     In this test we analyze the data to know if there is any relationship between the awareness level of 
                     EVs and the demographic informations of the people in Tamil Nadu
                     """)
            
            st.markdown("""
                        ## Topics under the Objective:
                        ### Checking if there is a relationship between-
                        - Awareness level of EVs and people's Gender
                        - Awareness level of EVs and people's Age Category
                        - Awareness level of EVs and people's Area of Residence
                        - Awareness level of EVs and people's Educational Qualification
                        - Awareness level of EVs and people's Employment Status
                        - Awareness level of EVs and Different types of Vehicle Users
                        - Awareness level of EVs and EV buyers and non-buyers
                        """)


            st.markdown("""
                        ## Fixing Hypothesis:

                        1. Awareness level of EVs based on people's gender
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and people's gender.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and people's gender.
                            
                        2. Awareness level of EVs based on people's age Category
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and people's age category.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and people's age category.
                        
                        3. Awareness level of EVs based on people's area of residence
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and people's area of residence.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and people's area of residence.
                            
                        4. Awareness level of EVs based on people's educational qualification
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and people's educational qualification.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and people's educational qualification.
                            
                        5. Awareness level of EVs based on people's employment status
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and people's employment status.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and people's employment status.
                            
                        6. Awareness level of EVs among different types of Vehicle Users
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and different types of vehicle users.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and different types of vehicle users.
                            
                        7. Awareness level of EVs among EV buyers and non-buyers
                            - **Null Hypothesis**: There is no significant relationship between the awareness level of EVs and EV buyers and non-buyers.
                            - **Alternative Hypothesis**: There is a significant relationship between the awareness level of EVs and EV buyers and non-buyers.
                            """)

            st.markdown("""
                        ### Chi-square Test Proceedure:
                        - 1. **Fix the Hypothesis**:
                            - **H0**: No relationship between the variables.
                            - **H1**: A relationnship between the variables.
                        - 2. **Select (α)** = 0.05.
                        - 3. **Choose a test**: Chi-square tes for independence.
                        - 4. **Collect data**: Any survey sample data.
                        - 5. **Calculate the test statistic**: Determine the Chi-sqaure statistic from data.
                        - 6. **Determine the p-value**: Find the p-value associated with the calculated statistic.
                        - 7. **Make a decision**: If p < 0.05, reject H0; otherwise fail to reject H0.
                        - 8. **Draw conlusion**: Intereptinng the results regardinng awareness levels.
                        """)

        elif opts == "Results":
            st.header("Results of Chi-square test")
            st.write("""
                     ### After Performing Chi-square Test for Independence of Attributes :

                     ### 1. Awareness Levels of EVs Based on Gender:
                     After performing the Chi-Square test of independence, the p-value was greater than 0.05. 
                     
                     **Conclusion**: We fail to reject the null hypothesis. This means that there is 
                     no significant relationship between the awareness level of EVs and people's 
                     gender. The awareness level of electric vehicles does not significantly differ 
                     by gender in our dataset.

                     ### 2. Awareness Levels of EVs based on Age Category:
                     After performing the Chi-Square test of independence, the p-value was greater than 0.05. 
                     
                     **Conclusion**: We fail to reject the null hypothesis. This means that there is no 
                     significant relationship between the awareness level of EVs and people's age category. 
                     The awareness level of electric vehicles does not significantly differ 
                     by people's age category in our dataset.

                     ### 3. Awareness Levels of EVs based on Area of Residence:
                     After performing the Chi-Square test of independence, the p-value was found to be less than 0.05.
                     
                     **Conclusion**: We reject the null hypothesis and accept the alternative hypothesis. 
                     There is a significant relationship between the awareness level of EVs and people's 
                     area of residence. Area of residence may influence how aware individuals 
                     are of electric vehicles.

                     ### 4. Awareness Levels of EVs based on Educational Qualification:
                     After performing the Chi-Square test of independence, the p-value was greater than 0.05. 
                     
                     **Conclusion**: We fail to reject the null hypothesis. This means that there is no 
                     significant relationship between the awareness level of EVs and people's 
                     educational qualification. The awareness level of electric vehicles does not 
                     significantly differ by people's educational qualification in our dataset.


                     ### 5. Awareness Levels of EVs based on Employment Status:
                     After performing the Chi-Square test of independence, the p-value was greater than 0.05. 
                     
                     **Conclusion**: We fail to reject the null hypothesis. This means that there is no 
                     significant relationship between the awareness level of EVs and people's 
                     employment status. The awareness level of electric vehicles does not significantly 
                     differ by people's employment in our dataset.


                     ### 6. Awareness Levels of EVs based on Different Types of Vehicle Users:
                     After performing the Chi-Square test of independence, the p-value was found to be less than 0.05.
                     
                     **Conclusion**: We reject the null hypothesis and accept the alternative hypothesis. 
                     There is a significant relationship between the awareness level of EVs and  
                     different Types of Vehicle Users. Different Types of Vehicle Users may influence 
                     how aware individuals are of electric vehicles.

                     ### 7. Awareness Levels of EVs based on EV Buyers and Non-Buyers:
                     After performing the Chi-Square test of independence, the p-value was found to be less than 0.05.
                     
                     **Conclusion**: We reject the null hypothesis and accept the alternative hypothesis. 
                     There is a significant relationship between the awareness level of EVs and 
                     EV buyers and non-buyers. EV buyers and non-buyers may influence how aware 
                     individuals are of electric vehicles.
                     """)
        if opts == "Final Conclusion":
            st.write("""
                     ## Final Conclusion:

                     After performing the Chi-Square test for independence to examine the relationship between **Awareness Levels of Electric Vehicles (EVs)** and various demographic factors, we observed the following:

                     1. **Gender**: The analysis revealed no significant relationship between gender and awareness levels of EVs. This indicates that people's awareness of electric vehicles is not influenced by their gender in our dataset.
                    
                     2. **Age Category**: Similarly, there was no significant relationship between age category and awareness levels of EVs. This suggests that people's awareness of electric vehicles does not vary significantly across different age groups.

                     3. **Area of Residence**: The test showed a significant relationship between area of residence and awareness levels of EVs. This suggests that people living in different areas have varying levels of awareness about electric vehicles. Therefore, where a person lives may play an important role in how informed they are about EVs.

                     4. **Educational Qualification**: No significant relationship was found between educational qualification and awareness levels of EVs. This means that people with different levels of education exhibit similar awareness about electric vehicles.

                     5. **Employment Status**: The results indicated no significant relationship between employment status and awareness levels of EVs. Thus, being employed or unemployed does not significantly impact how aware individuals are about electric vehicles.

                     6. **Different Types of Vehicle Users**: A significant relationship was found between different types of vehicle users and awareness levels of EVs. This suggests that the type of vehicle a person uses has an influence on how aware they are of electric vehicles, potentially due to greater exposure or relevance.

                     7. **EV Buyers and Non-Buyers**: There was a significant relationship between being an EV buyer or non-buyer and the awareness levels of EVs. This indicates that those who have purchased EVs, or have considered purchasing them, are more aware of electric vehicles compared to those who haven't.

                     ## Overall Insights:

                     From these findings, we can conclude that **factors such as area of residence, type of vehicle usage, and purchasing behavior (buyer vs. non-buyer)** significantly influence awareness levels of electric vehicles. On the other hand, **gender, age, education, and employment status** do not appear to play a significant role in EV awareness.

                     These insights can guide future awareness campaigns or policy efforts to target specific demographics, especially in areas and among vehicle users with lower awareness. Additionally, increasing EV ownership and exposure could enhance overall awareness levels.
                     
                     This information could be valuable for marketing strategies aimed at increasing 
                     EV adoption, as it highlights the immportance of targeting potential buyers with 
                     effective awareness campaigns.
                     """)
 


    with tab2:

        st.title("Predictive Systems Created Using Streamlit")
        if st.button("View Raw Files"):
            webbrowser.open_new_tab("https://github.com/Anand-krish-0/q_l_project/tree/main/ML_project_streamlit")

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
            st.markdown("Model preperation:")
            st.success("The Lasso Regression model achieved an accuracy of 87.09% in prediction.")
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
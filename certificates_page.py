import streamlit as st
import base64

def certificates():
    st.title("Course Certificates")
    st.write("Here are some of my completed course certificates and you can view them below.")
    
    # Creating two tabs for Python and SQL courses
    tab1, tab2 = st.tabs(["Python Courses", "SQL Courses"])

    # Content for the "Python Courses" tab
    with tab1:
        st.header("Python Courses")

        # Display Python for Data Science certificate (Image)
        st.subheader("1. Python for Data Science")
        data_science_image = r"files/Python for Data Science_certificate (1).jpg"
        st.image(data_science_image, caption="Python for Data Science", use_column_width=True)

        # Display Pandas Course certificate (Image)
        st.subheader("2. Pandas Course")
        pandas_image = r"files/pandas course_page-0001.jpg"
        st.image(pandas_image, caption="Pandas Course", use_column_width=True)

        # Display Numpy Course certificate (Image)
        st.subheader("3. Numpy Course")
        numpy_image = r"files/numpy course.jpg"
        st.image(numpy_image, caption="Numpy Course", use_column_width=True)

    # Content for the "SQL Courses" tab
    with tab2:
        st.header("SQL Courses")

        # Display Introduction to SQL certificate (Image)
        st.subheader("1. Introduction to SQL")
        sql_intro_image = r"files/Introduction to SQL_certificate (1).jpg"
        st.image(sql_intro_image, caption="Introduction to SQL", use_column_width=True)

        # Display SQL Intermediate certificate (Image)
        st.subheader("2. SQL Intermediate")
        sql_inter_image = r"files/SQL Intermediate_certificate.jpg"
        st.image(sql_inter_image, caption="SQL Intermediate", use_column_width=True)


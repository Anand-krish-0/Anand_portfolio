import streamlit as st
import base64

def certificates():

    st.title("Course Certificates")
    st.write("Here are some of my completed course certificates. You can view them below and download if you'd like.")

    st.header("Python Courses")

    def show_pdf(file_path):
        # Embed the PDF in an iframe with toolbar disabled
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0" width="700" height="550" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


    # Display Python for Data Science certificate (Image)
    st.subheader("1. Python for Data Science")
    data_science_image = r"files\Python for Data Science_certificate (1).jpg"  # Corrected path formatting
    st.image(data_science_image, caption="Python for Data Science", use_column_width=True)

    # Display Pandas Course certificate (PDF)
    st.subheader("2. Pandas Course")
    pdf_file1 = r"files/pandas course.pdf"  # Ensure path uses forward slashes or raw string
    show_pdf(pdf_file1)

    # Display Numpy Course certificate (PDF)
    st.subheader("3. Numpy Course")
    pdf_file2 = r"files/numpy course.pdf"  # Ensure path uses forward slashes or raw string
    show_pdf(pdf_file2)

    # SQL Courses
    st.header("SQL Courses")

    # Display Introduction to SQL certificate (Image)
    st.subheader("1. Introduction to SQL")
    sql_intro_image = r"files/Introduction to SQL_certificate (1).jpg"  # Corrected path formatting
    st.image(sql_intro_image, caption="Introduction to SQL", use_column_width=True)

    # Display SQL Intermediate certificate (Image)
    st.subheader("2. SQL Intermediate")
    sql_inter_image = r"files/SQL Intermediate_certificate.jpg"  # Corrected path formatting
    st.image(sql_inter_image, caption="SQL Intermediate", use_column_width=True)



import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def clusterpage():

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

    st.header("Principal Component Analysis(PCA) and Clustering")
    st.subheader("Clustered Datapoints")
    image_path = "images\cluster_image.png"
    img_str = load_image(image_path)
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
    st.header("Hierarchical Agglomerative Clustering (HAC)")
    # Display section header
    st.subheader("Cluster Dendogram")

    # Specify the image path (use forward slashes)
    image1_path = "images/cluster_image.png"  # Corrected path

    # Load the image
    img_str1 = load_image(image1_path)

    # Display the image if it was loaded successfully
    if img_str1:
        st.image(f"data:image/png;base64,{img_str1}", caption="Clustered Dendogram", use_column_width=True)

    # Displaying the Clustering section in one st.write
    st.write("""
            #### Role of Hierarchical Agglomerative Clustering (HAC)

            **Purpose**: Hierarchical Agglomerative Clustering (HAC) is an unsupervised learning technique used to group 
            similar data points based on their features. In this project, HAC helps identify distinct user segments 
            based on awareness and perception of electric vehicles (EVs).

            #### Key Contributions:

            - **User Segmentation**: HAC groups users into clusters, revealing patterns in their attitudes towards EVs.

            - **Visual Insights**: The dendrogram produced by HAC visually represents the relationships between clusters, 
            aiding in understanding user similarities and differences.

            - **Targeted Strategies**: By analyzing each cluster's characteristics, stakeholders can develop tailored 
            interventions, such as awareness campaigns or financial incentives, to address specific user needs and enhance EV adoption.
            """)
    
    st.subheader("Clustered Datapoints (2D)")

    # Specify the image path (use forward slashes)
    image2_path = "images/cluster_points_2D.png"  # Corrected path

    # Load the image
    img_str2 = load_image(image2_path)

    # Display the image if it was loaded successfully
    if img_str2:
        st.image(f"data:image/png;base64,{img_str2}", caption="Clustered Data", use_column_width=True)
            

    st.subheader("Clustered Datapoits (3D)")

    # Specify the image path (use forward slashes)
    image3_path = "images/cluster_points_3D.png" 

    # Load the image
    img_str3 = load_image(image3_path)

    # Display the image if it was loaded successfully
    if img_str3:
        st.image(f"data:image/png;base64,{img_str3}", caption="Clustered Data", use_column_width=True) 
    
    # Displaying the Cluster Analysis Results and Insights in one st.write
    st.write("""
            ### Cluster Analysis Results:

            - **Cluster 0**: 126 users - Likely high awareness and positive perceptions of EVs.
            - **Cluster 1**: 70 users - Moderate awareness and mixed feelings about EVs.
            - **Cluster 2**: 5 users - Low awareness or negative perceptions of EVs.

            ### Cluster Analysis and Insights:

            #### 1. Cluster Distribution:

            - **Cluster 0 (126 users)**: This cluster represents the majority of your respondents. They may have a high 
            level of awareness and positive perception regarding electric vehicles. Further 
            analysis of their demographics (e.g., age, education, employment status) could 
            provide insights into what drives their favorable perception.

            - **Cluster 1 (70 users)**: This group has moderate awareness or mixed feelings about EVs. Analyzing the user 
            satisfaction and cost consideration features for this cluster could highlight areas 
            that need improvement to enhance their perception of EVs.

            - **Cluster 2 (5 users)**: This small cluster likely includes individuals with low awareness or negative 
            perceptions about EVs. Understanding their characteristics and concerns can help 
            develop targeted interventions to improve their attitudes.

            #### Demographic Factors:
            Analyzing the demographic features (like gender, age, employment status, and education) in relation to 
            each cluster can reveal patterns. For example, you may find that younger individuals or those with 
            higher education levels are more represented in Cluster 0, indicating a correlation between education and 
            awareness of EVs.

            #### Current Awareness:
            The features related to "current awareness" (1.1 to 1.4) can be analyzed to determine how 
            awareness levels vary among clusters. If Cluster 0 scores significantly higher in these 
            features, it confirms their positive attitude towards EVs. For Clusters 1 and 2, 
            identifying which aspects of awareness are lacking can guide educational initiatives.

            #### User Satisfaction:
            The user satisfaction features (2.1 to 2.5) can be assessed to gauge the level of 
            contentment with current EV options. If Clusters 1 and 2 show lower satisfaction scores, 
            it may indicate issues with vehicle performance, support services, or user experience, 
            providing a focus area for manufacturers and policymakers.

            #### Charging Facilities:
            Features related to charging facilities (3.1 to 3.4) are critical for EV adoption. Analyzing 
            responses can reveal whether accessibility to charging infrastructure is a barrier for 
            Clusters 1 and 2. Users in these clusters might express concerns about the availability or 
            convenience of charging stations.

            #### Cost Considerations:
            Assessing the cost-related features (4.1 to 4.5) can provide insights into how financial 
            factors influence user perceptions. If Clusters 1 and 2 have significant concerns about 
            cost, this suggests a need for financial incentives or more affordable EV options to 
            encourage adoption.

            #### Consumer Preferences:
            Understanding preferences (5.1 to 5.5) regarding vehicle types, brands, and features among 
            different clusters can help tailor marketing strategies. Cluster 0 might prioritize features 
            like sustainability, while Clusters 1 and 2 might have more traditional preferences that 
            need to be addressed.

            #### EVM Considerations:
            Features related to electric vehicle maintenance (6.1 to 6.4) should be examined to 
            understand how maintenance perceptions influence overall satisfaction and awareness. 
            If Clusters 1 and 2 show concerns about maintenance, manufacturers might consider addressing 
            these in their communication.

            ### Conclusions and Recommendations:
            - **Targeted Awareness Campaigns**: Based on the findings, targeted awareness campaigns could be 
            developed to address the specific concerns and knowledge gaps identified in Clusters 1 and 2. 
            This can help in improving overall perceptions of electric vehicles.

            - **Enhancing User Experience**: Gathering feedback from Clusters 1 and 2 can lead to improvements in user satisfaction, 
            focusing on aspects like charging facilities, customer service, and overall vehicle experience.

            - **Policy Implications**: Policymakers should consider financial incentives for EV adoption and the development of 
            charging infrastructure to alleviate concerns related to costs and accessibility, 
            particularly for users in Clusters 1 and 2.

            - **Further Research**: Future studies could explore qualitative aspects through interviews or focus groups to 
            deepen the understanding of users' perceptions and barriers to EV adoption.
             
             #### Conclusion:
            HAC plays a crucial role in uncovering insights about user perceptions of EVs, facilitating informed 
            decision-making to promote electric vehicle adoption in Tamil Nadu.
            """)

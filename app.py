import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved K-Means model and scaler
kmeans = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler_model.joblib')

# Define the feature names used in the clustering model
features = [
    'Birth Rate', 'CO2 Emissions', 'Days to Start Business', 'GDP',
    'Health Exp % GDP', 'Health Exp/Capita', 'Infant Mortality Rate',
    'Internet Usage', 'Lending Interest', 'Life Expectancy Female',
    'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14',
    'Population 15-64', 'Population 65+', 'Population Total',
    'Population Urban', 'Tourism Inbound', 'Tourism Outbound'
]

# Map the cluster number to its name based on analysis
cluster_names = {
    0: "Developed Economies",
    1: "High-Income Economies",
    2: "Low-Income Economies"
}

# App title with custom styling
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Global Development Clustering</h1>", unsafe_allow_html=True)
st.markdown("""
### Explore how different countries are categorized based on economic and social factors.
Input feature values and see which cluster your country fits into.
""")

# Sidebar for user input with enhanced styling
st.sidebar.header("Input Features")
user_input = []

# Use number input for each feature to allow manual input, with default value set to 0.0
for feature in features:
    value = st.sidebar.number_input(
        f"Enter {feature}",
        min_value=0.0,
        value=0.0,  # Default value for all features is set to 0.0
        step=0.1
    )
    user_input.append(value)

# Convert user inputs into a NumPy array for scaling
user_data = np.array(user_input).reshape(1, -1)

# Check if all values are zero
if np.all(user_data == 0):
    st.error("Error: All input values are zero. Please enter valid non-zero values.")
else:
    input_scaled = scaler.transform(user_data)

    # Predict cluster on button click
    if st.button("Predict Cluster"):
        # Predict the cluster
        cluster = kmeans.predict(input_scaled)
        cluster_name = cluster_names.get(cluster[0], "Unknown Cluster")
        st.success(f"Your country is classified into the **{cluster_name}** cluster!")

        # Bar chart for the user's input values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=user_input, y=features, palette="Blues_d", ax=ax)
        ax.set_xlabel("Feature Value", fontsize=14, color='#3b3b3b')
        ax.set_title("Features", fontsize=16, color='#FF6347')
        plt.tight_layout()
        st.pyplot(fig)

# Display some additional info in a clean and styled way
st.markdown("""
### Cluster Descriptions:
- **Developed Economies**: High GDP, excellent healthcare, and high life expectancy.
- **High-Income Economies**: Extremely high GDP, strong healthcare, and advanced technology.
- **Low-Income Economies**: Lower GDP, less access to healthcare, and shorter life expectancy.
""")

st.write("This app is powered by K-Means clustering, which groups countries into distinct categories based on various economic and social factors, allowing you to explore and predict the development classification of any given country")
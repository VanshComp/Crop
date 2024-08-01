import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(
    page_title="SmartCrop", 
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png", 
    layout='centered', 
    initial_sidebar_state="collapsed"
)

def load_model(modelfile):
    """Load and return the machine learning model."""
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Load and prepare crop recommendation data
PATH = './Crop_recommendation.csv'
data = pd.read_csv(PATH)

# Prepare features and labels
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

for model_name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    models[model_name] = {'model': model, 'accuracy': accuracy}

# Find the best model
best_model_name = max(models, key=lambda name: models[name]['accuracy'])
best_model = models[best_model_name]['model']

# Crop Rotation Planner Class
class CropRotationPlanner:
    def __init__(self, db):
        self.db = db

    def plan_rotation(self, field_history):
        if not field_history:
            raise ValueError("Field history cannot be empty")

        last_crop = field_history[-1]
        if last_crop not in self.db:
            raise ValueError(f"Last crop '{last_crop}' is not in the database")

        last_crop_family = self.db[last_crop]['family']
        current_season = self.get_current_season()

        eligible_crops = [crop for crop, info in self.db.items() if info['season'] == current_season]
        eligible_crops = [crop for crop in eligible_crops if self.db[crop]['family'] != last_crop_family]

        return eligible_crops

    def get_current_season(self):
        # Placeholder for actual implementation
        return 'spring'

# Crop database with more crops added
db = {
    'wheat': {'season': 'spring', 'family': 'grass'},
    'barley': {'season': 'spring', 'family': 'grass'},
    'oats': {'season': 'spring', 'family': 'grass'},
    'soybeans': {'season': 'spring', 'family': 'legume'},
    'corn': {'season': 'spring', 'family': 'grass'},
    'sunflowers': {'season': 'spring', 'family': 'composite'},
    'carrots': {'season': 'spring', 'family': 'parsley'},
    'lettuce': {'season': 'spring', 'family': 'composite'},
    'tomatoes': {'season': 'summer', 'family': 'nightshade'},
    'potatoes': {'season': 'summer', 'family': 'nightshade'},
    'cucumbers': {'season': 'summer', 'family': 'cucurbit'},
    'beets': {'season': 'spring', 'family': 'goosefoot'},
    'peas': {'season': 'spring', 'family': 'legume'},
    'radishes': {'season': 'spring', 'family': 'mustard'},
    'onions': {'season': 'spring', 'family': 'amaryllis'}
}

planner = CropRotationPlanner(db)

# Streamlit App
st.title("SmartCrop: Intelligent Crop Recommendation and Rotation Planner")

# Visualization Section
st.header("Data Insights and Visualizations")

# Plot histograms for features
st.subheader("Distribution of Features")
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
for i, feat in enumerate(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']):
    sns.histplot(data[feat], kde=True, ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(f'Distribution of {feat}')
plt.tight_layout()
st.pyplot(fig)

# Plot correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis', ax=ax)
ax.set_title('Correlation between Features')
st.pyplot(fig)

# Crop Rotation Planner Section
st.header("Crop Rotation Planner")
field_history = st.text_input("Enter field history (comma-separated)", "wheat,soybeans,corn").split(',')
field_history = [crop.strip() for crop in field_history]

if st.button("Plan Rotation"):
    try:
        eligible_crops = planner.plan_rotation(field_history)
        st.success(f"Eligible crops for the next season: {', '.join(eligible_crops)}")
    except ValueError as e:
        st.error(e)

# Crop Recommendation System Section
st.header("Crop Recommendation System")
st.write("Enter the input features to get the recommended crop.")

N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, step=1.0)
P = st.number_input("Phosphorous (P)", min_value=0.0, max_value=150.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, step=1.0)
temperature = st.number_input("Temperature", min_value=0.0, max_value=100.0, step=1.0)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, step=1.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall", min_value=0.0, max_value=300.0, step=1.0)

if st.button("Predict"):
    new_data_values = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = best_model.predict(new_data_values)
    st.success(f"The predicted crop label is {prediction[0]}")

# Hide the Streamlit menu and adjust layout
hide_menu_style = """
    <style>
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

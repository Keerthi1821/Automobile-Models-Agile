import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Setting page config at the very beginning
st.set_page_config(page_title="Car Insights Dashboard", layout="wide")

# Load the data
@st.experimental_singleton
def load_data():
    data = pd.read_csv(r"C:\Users\keert\Downloads\CarsData.csv")

    return data

data = load_data()

# Preprocessing and model setup
categorical_features = ['transmission', 'fuelType']
one_hot = OneHotEncoder()
preprocessor = ColumnTransformer(transformers=[
    ('cat', one_hot, categorical_features)],
    remainder='passthrough')
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Splitting the dataset
X = data[['year', 'mileage', 'engineSize', 'transmission', 'fuelType']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

# Streamlit layout enhancements
st.title('Car Price Prediction and Performance Analysis')

# User inputs for prediction in sidebar
st.sidebar.header('Predict Car Prices')
year = st.sidebar.number_input('Car Year', min_value=int(X['year'].min()), max_value=int(X['year'].max()), value=2017)
mileage = st.sidebar.number_input('Mileage', min_value=0, value=5000)
engine_size = st.sidebar.number_input('Engine Size', min_value=0.0, max_value=5.0, value=1.0)
transmission = st.sidebar.selectbox('Transmission', options=X['transmission'].unique())
fuel_type = st.sidebar.selectbox('Fuel Type', options=X['fuelType'].unique())

if st.sidebar.button('Predict Price'):
    input_df = pd.DataFrame([[year, mileage, engine_size, transmission, fuel_type]],
                            columns=['year', 'mileage', 'engineSize', 'transmission', 'fuelType'])
    prediction = model.predict(input_df)
    st.sidebar.success(f'The predicted price of the car is ${prediction[0]:,.2f}')

# Plotting
col1, col2 = st.columns(2)
with col1:
    st.header('Price vs Mileage by Manufacturer')
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='mileage', y='price', hue='Manufacturer', ax=ax)
    st.pyplot(fig)

with col2:
    st.header('Average Price by Manufacturer')
    avg_price_data = data.groupby('Manufacturer')['price'].mean().sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=avg_price_data.values, y=avg_price_data.index, ax=ax)
    st.pyplot(fig)

st.header('Mileage Distribution by Car Model')
fig, ax = plt.subplots(figsize=(10, 8))
selected_manufacturer = st.selectbox('Select Manufacturer', options=data['Manufacturer'].unique())
filtered_data = data[data['Manufacturer'] == selected_manufacturer]
sns.boxplot(data=filtered_data, x='mileage', y='model', ax=ax)
st.pyplot(fig)

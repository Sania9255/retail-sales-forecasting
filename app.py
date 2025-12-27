import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/rf_model.pkl")

# Load processed data for reference
df = pd.read_csv('data/processed/processed.csv')

st.title("🛒 Walmart Weekly Sales Forecasting")

st.write("Predict Weekly Sales based on Store, Department, and other features")

# User inputs
store = st.number_input("Store Number", min_value=int(df['Store'].min()), max_value=int(df['Store'].max()), value=1)
dept = st.number_input("Department Number", min_value=int(df['Dept'].min()), max_value=int(df['Dept'].max()), value=1)
size = st.number_input("Store Size", value=int(df['Size'].mean()))
type_map = {"A": 1, "B": 2, "C": 3}
store_type = st.selectbox("Store Type", options=["A","B","C"])
type_encoded = type_map[store_type]
temperature = st.number_input("Temperature", value=float(df['Temperature'].mean()))
fuel_price = st.number_input("Fuel Price", value=float(df['Fuel_Price'].mean()))
cpi = st.number_input("CPI", value=float(df['CPI'].mean()))
unemployment = st.number_input("Unemployment", value=float(df['Unemployment'].mean()))
is_holiday = st.checkbox("Is Holiday?")

# Prepare input DataFrame
input_df = pd.DataFrame({
    "Store": [store],
    "Dept": [dept],
    "Size": [size],
    "Type": [type_encoded],
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "IsHoliday": [is_holiday]
})

# Predict buttonimport streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Load model
model = joblib.load("models/rf_model.pkl")

# Load data
df = pd.read_csv("data/processed/processed.csv")

st.title("🛒 Retail Weekly Sales Forecasting Dashboard")

st.markdown("Predict **Weekly Sales** using Machine Learning and visualize trends")

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("Enter Store Details")

store = st.sidebar.number_input(
    "Store Number",
    int(df["Store"].min()),
    int(df["Store"].max()),
    int(df["Store"].mode()[0])
)

dept = st.sidebar.number_input(
    "Department Number",
    int(df["Dept"].min()),
    int(df["Dept"].max()),
    int(df["Dept"].mode()[0])
)

size = st.sidebar.number_input("Store Size", int(df["Size"].mean()))

store_type = st.sidebar.selectbox("Store Type", ["A", "B", "C"])
type_map = {"A": 1, "B": 2, "C": 3}
type_encoded = type_map[store_type]

temperature = st.sidebar.slider(
    "Temperature",
    float(df["Temperature"].min()),
    float(df["Temperature"].max()),
    float(df["Temperature"].mean())
)

fuel_price = st.sidebar.slider(
    "Fuel Price",
    float(df["Fuel_Price"].min()),
    float(df["Fuel_Price"].max()),
    float(df["Fuel_Price"].mean())
)

cpi = st.sidebar.slider(
    "CPI",
    float(df["CPI"].min()),
    float(df["CPI"].max()),
    float(df["CPI"].mean())
)

unemployment = st.sidebar.slider(
    "Unemployment",
    float(df["Unemployment"].min()),
    float(df["Unemployment"].max()),
    float(df["Unemployment"].mean())
)

is_holiday = st.sidebar.checkbox("Is Holiday")

# ------------------ INPUT DATAFRAME ------------------
input_df = pd.DataFrame({
    "Store": [store],
    "Dept": [dept],
    "Size": [size],
    "Type": [type_encoded],
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "IsHoliday": [is_holiday]
})

# ------------------ PREDICTION ------------------
st.subheader("📊 Prediction Result")

if st.button("Predict Weekly Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Weekly Sales: ${prediction:,.2f}")

# ------------------ HISTORICAL SALES ------------------
st.subheader("📈 Historical Weekly Sales Trend")

sample_size = min(100, len(df))
sample_df = df.sample(sample_size).sort_index()

fig, ax = plt.subplots()
ax.plot(sample_df["Weekly_Sales"])
ax.set_xlabel("Weeks")
ax.set_ylabel("Weekly Sales")
ax.set_title("Historical Weekly Sales Trend")

st.pyplot(fig)

# ------------------ ACTUAL vs PREDICTED ------------------
st.subheader("📉 Actual vs Predicted Sales Comparison")

try:
    pred_df = pd.read_csv("data/processed/predictions.csv")
    sample_size = min(50, len(pred_df))
    pred_df = pred_df.head(sample_size)

    fig2, ax2 = plt.subplots()
    ax2.plot(pred_df["Actual"], label="Actual Sales")
    ax2.plot(pred_df["Predicted"], label="Predicted Sales")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Weekly Sales")
    ax2.legend()

    st.pyplot(fig2)

except Exception as e:
    st.warning("Prediction comparison data not available yet.")

# ------------------ DATA PREVIEW ------------------
st.subheader("🧾 Dataset Preview")
st.dataframe(df.head())

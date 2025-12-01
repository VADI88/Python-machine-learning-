# IMPORT


import pandas as pd
import plotly.express as px
import streamlit as st

# Getting the data

# sales_df= pd.read_csv("../data/sales_data.csv")

# Title and application

st.title("Business Dashboard")
st.write("""
This app provides insights into sales,customer demographics and product performance
""")

# 2.0 Data input

st.header("Upload Business Data")
uploaded_file = st.file_uploader(
    label="Choose a CSV File", type="csv", accept_multiple_files=False
)
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of the data")
    st.write(data.head())

    st.subheader("Sales Insights")
    if ("sales_date" in data.columns) and ("sales_amount" in data.columns):
        st.write("Sales over time")
        fig = px.line(
            data, x="sales_date", y="sales_amount", title="Sales over time"
        )
        st.plotly_chart(fig)
    else:
        st.warning(
            "Please ensure your data contains `sales_date` and `sales_amount` columns."
        )

    st.subheader("Customer segmentation")
    if "region" in data.columns and "sales_amount" in data.columns:
        st.write("Values by Region")
        fig = px.pie(
            data,
            names="region",
            values="sales_amount",
            title="Sales over region",
        )
        st.plotly_chart(fig)
    else:
        st.warning(
            "Please ensure your data contains `region` and `sales_amount` columns."
        )

    st.subheader("Product Analysis")
    if "product" in data.columns and "sales_amount" in data.columns:
        st.write("Sales by Product")
        top_product_df = (
            data.groupby(["product"], as_index=False)
            .sales_amount.sum()
            .nlargest(columns="sales_amount", n=10)
        )

        fig = px.bar(
            top_product_df,
            x="sales_amount",
            y="product",
            title="Top Products by Sales",
        )

        st.plotly_chart(fig)
    else:
        st.warning(
            "Please ensure your data contains `Product` and `sales_amount` columns."
        )

    st.header("Your feedback count!!")
    feedback = st.text_area("Please provide your feedback")

    if st.button("Submit"):
        st.text("Thank your feedback!")

# Footer
st.write("---")
st.write("This dashboards is flexible.Expand based on your feedback")

if __name__ == "__main__":
    pass

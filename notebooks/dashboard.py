import pandas as pd
import plotly.express as px
import streamlit as st

pd.set_option("display.max_columns", None)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = (df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_"))
    for col in df.columns:
        if col.endswith("date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter

    return df


def apply_filter(df):
    with st.sidebar:

        st.title("Filters")
        filter_columns = ["year", "region", "state", "segment", "category", "sub_category", "ship_mode"]

        for col in filter_columns:
            selected = st.multiselect(f"Selection {col.title()}", sorted(df[col].unique()))
            if selected:
                df = df[df[col].isin(selected)]

        profit_status = st.selectbox("Profit Status", ["All", "Profit", "Loss"])
        if profit_status == "Profit":
            df = df[df["profit"] > 0]
        elif profit_status == "Loss":
            df = df[df["profit"] < 0]
    return df


def show_metrics(df):
    total_profit = df["profit"].sum()
    total_sales = df["sales"].sum()
    total_orders = df["order_id"].nunique()
    aov = total_sales / total_orders
    profit_margin = (total_profit / total_sales) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Profit", f"{total_profit:,.0f}$")
    col2.metric("Total Sales", f"{total_sales:,.0f}$")
    col3.metric("Orders", total_orders)
    col4.metric("Average Order Value (AOV)", f"{aov:,.0f}$")
    col5.metric("Profit Margin", f"{profit_margin:.2f}%")


def show_monthly_sales(df):
    monthly_sales = (df.groupby(pd.Grouper( key="order_date",freq="ME"))["sales"].sum().reset_index())
    fig = px.line(monthly_sales, x="order_date", y="sales",markers=True, title="Monthly total Sales")
    st.plotly_chart(fig, use_container_width=True)

def show_region_sales(df):
    region_sales = df.groupby("region")["sales"].sum().reset_index()
    fig = px.bar(region_sales, x="region", y="sales", title="total Sales by Region")
    st.plotly_chart(fig, use_container_width=True)

def show_city_sales(df):
    city_sales = df.groupby("city")["sales"].sum().nlargest(15).reset_index()
    fig = px.bar(city_sales, x='city', y='sales', title="Top 15 Cities by Total Sales")
    st.plotly_chart(fig, use_container_width=True)

def show_category_profit(df):
    category_profit = (df.groupby("category")["profit"].sum().reset_index())
    fig = px.bar(category_profit, x="category", y="profit", color="category", title="Profit by Category")
    st.plotly_chart(fig, use_container_width=True)


def show_top_customers(df):
    customers = (df.groupby("customer_name")["sales"].sum().sort_values(ascending=False).reset_index())
    top_customers = customers[:50]
    fig = px.bar(top_customers, x="sales", y="customer_name", orientation="h",title="Top Customers")
    st.plotly_chart(fig, use_container_width=True)


def show_orders_trend(df):
    orders = (df.groupby(pd.Grouper(key="order_date", freq="ME"))["order_id"].nunique().reset_index())
    fig = px.line(orders, x="order_date", y="order_id", markers=True, title="Orders Trend")
    st.plotly_chart(fig, use_container_width=True)


def show_sales_profit(df):
    fig = px.scatter(df, x="sales", y="profit", color="category", size="quantity", hover_name="product_name", title="Sales vs Profit")
    st.plotly_chart(fig, use_container_width=True)

def show_top_products(df):
    top_products = (df.groupby("product_name")["sales"].sum().nlargest(10).reset_index())
    fig = px.bar(top_products, x="sales", y="product_name", orientation="h", title="Top 10 Products")
    st.plotly_chart(fig, use_container_width=True)

def show_worst_products(df):
    products = (df.groupby("product_name")["profit"].sum().sort_values())
    loss_products = products.head(50)
    st.subheader("Worst Products")
    st.dataframe(loss_products, use_container_width=True)



if __name__ == "__main__":
    st.set_page_config("Superstore Dashboard", layout='wide')
    st.title("Superstore Dashboard")

    df = load_data("./datasets/Superstore.csv")   
     
    df = apply_filter(df)
    show_metrics(df)
    st.dataframe(df)

    show_monthly_sales(df)
    show_region_sales(df)
    show_city_sales(df)
    show_category_profit(df)
    show_top_customers(df)
    show_orders_trend(df)
    show_sales_profit(df)
    show_top_products(df)
    show_worst_products(df)


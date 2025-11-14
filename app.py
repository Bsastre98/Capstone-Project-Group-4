import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import shapiro, skew, kurtosis, linregress

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="BMW Sales Analysis – Fast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# DATA LOADING + FEATURE ENGINEERING
# -------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("BMW Data Final Project.csv")
    return df


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_analysis = df.copy()

    # Revenue
    df_analysis["Revenue"] = df_analysis["Price_USD"]

    # Vehicle Category
    def categorize_vehicle(model):
        if model in ["X1", "X3", "X5", "X6"]:
            return "SUV"
        elif model in ["3 Series", "5 Series", "7 Series"]:
            return "Sedan"
        elif model in ["M2", "M3", "M4", "4 Series"]:
            return "Performance/Luxury"
        else:
            return "Other"
    df_analysis["Vehicle_Category"] = df_analysis["Model"].apply(categorize_vehicle)

    # Car Age
    current_year = int(df_analysis["Year"].max())
    df_analysis["Car_Age"] = current_year - df_analysis["Year"]

    # Price Category
    def categorize_price(p):
        if p < 30000:
            return "Budget"
        elif p < 50000:
            return "Mid-Range"
        else:
            return "Premium"
    df_analysis["Price_Category"] = df_analysis["Price_USD"].apply(categorize_price)

    # Mileage Category
    def categorize_mileage(m):
        if m < 20000:
            return "Low"
        elif m < 60000:
            return "Medium"
        elif m < 100000:
            return "High"
        else:
            return "Very High"
    df_analysis["Mileage_Category"] = df_analysis["Mileage_KM"].apply(categorize_mileage)

    # EV / Hybrid
    df_analysis["Is_EV_Hybrid"] = df_analysis["Fuel_Type"].isin(["Electric", "Hybrid"]).astype(int)

    # Performance score
    df_analysis["Performance_Score"] = (
        df_analysis["Engine_Size_L"] * 100
        + (df_analysis["Transmission"] == "Automatic").astype(int) * 50
        + df_analysis["Is_EV_Hybrid"] * 75
    )

    # Price per liter
    df_analysis["Price_per_Liter"] = df_analysis["Price_USD"] / df_analysis["Engine_Size_L"]

    # Rough depreciation
    df_analysis["Estimated_Depreciation_Rate"] = df_analysis["Car_Age"] * 0.15

    return df_analysis


# -------------------------------------------------
# EDA: STREAMLIT-FRIENDLY VISUALS
# -------------------------------------------------
def eda_distributions(df: pd.DataFrame):
    st.markdown("### 3.1 – Distribution Analysis")

    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    col = st.selectbox("Select numeric variable", numeric_cols, index=0)

    fig = px.histogram(
        df,
        x=col,
        nbins=50,
        title=f"{col.replace('_', ' ')} Distribution",
    )
    fig.update_traces(marker_line_color="black", marker_line_width=0.5)
    fig.update_layout(yaxis_title="Frequency", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Mean: **{df[col].mean():,.2f}** | Median: **{df[col].median():,.2f}** "
        f"| Min: **{df[col].min():,.2f}** | Max: **{df[col].max():,.2f}**"
    )


def eda_normality(df: pd.DataFrame):
    st.markdown("### 3.2 – Normality Tests")

    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    results = []

    for col in numeric_cols:
        data = df[col].dropna()
        stat, p_sw = shapiro(data)
        sk = skew(data)
        ku = kurtosis(data)
        results.append(
            {
                "Variable": col,
                "Shapiro-Wilk p-value": p_sw,
                "Is Normal (p>0.05)": "Yes" if p_sw > 0.05 else "No",
                "Skewness": sk,
                "Kurtosis": ku,
            }
        )
    normality_df = pd.DataFrame(results)

    st.dataframe(
        normality_df.style.format(
            {
                "Shapiro-Wilk p-value": "{:.4f}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.caption("p < 0.05 ⇒ variable is not normally distributed.")


def eda_relationships(df: pd.DataFrame):
    st.markdown("### 3.3 – Relationship Analysis")

    numerical_features = ["Price_USD", "Sales_Volume", "Engine_Size_L", "Mileage_KM", "Year"]
    corr_matrix = df[numerical_features].corr()

    st.markdown("#### Correlation Matrix")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Price vs Sales Volume by Region")
    fig_scatter = px.scatter(
        df,
        x="Price_USD",
        y="Sales_Volume",
        color="Region",
        hover_data=["Model", "Year"],
        title="Price vs Sales Volume by Region",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def eda_categorical(df: pd.DataFrame):
    st.markdown("### 3.4 – Categorical Variable Analysis")

    cat_vars = ["Model", "Region", "Fuel_Type", "Transmission", "Color", "Sales_Classification"]
    var = st.selectbox("Select categorical variable", cat_vars, index=1)

    grouped = (
        df.groupby(var)["Sales_Volume"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Sales_Volume": "Avg_Sales_Volume"})
    )

    fig = px.bar(
        grouped,
        x=var,
        y="Avg_Sales_Volume",
        title=f"Average Sales Volume by {var}",
        text_auto=".0f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Top category: **{grouped.iloc[0][var]}** "
        f"with average sales volume **{grouped.iloc[0]['Avg_Sales_Volume']:.0f}**"
    )


def eda_outliers(df: pd.DataFrame):
    st.markdown("### 3.5 – Outlier Detection")

    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    col = st.selectbox("Select variable for box plot", numeric_cols, index=0)

    fig = px.box(
        df,
        y=col,
        points="outliers",
        title=f"Outlier Detection for {col}",
    )
    st.plotly_chart(fig, use_container_width=True)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_pct = len(outliers) / len(df) * 100

    st.caption(
        f"Outliers: **{len(outliers)}** ({outlier_pct:.2f}% of records)  |  "
        f"IQR bounds: [{lower:,.2f}, {upper:,.2f}]"
    )


# -------------------------------------------------
# RQ1 – KEY FACTORS (CACHED MODELS + PLOTLY VISUALS)
# -------------------------------------------------
@st.cache_resource
def train_rq1_models(df_analysis: pd.DataFrame):
    df_enc = df_analysis.copy()

    categorical_cols = [
        "Model",
        "Region",
        "Color",
        "Fuel_Type",
        "Transmission",
        "Vehicle_Category",
        "Price_Category",
        "Mileage_Category",
    ]
    for col in categorical_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col + "_Encoded"] = le.fit_transform(df_enc[col].astype(str))

    numeric_base = [
        "Price_USD",
        "Engine_Size_L",
        "Mileage_KM",
        "Car_Age",
        "Performance_Score",
        "Is_EV_Hybrid",
        "Price_per_Liter",
    ]

    feature_cols = []
    feature_cols.extend([c for c in numeric_base if c in df_enc.columns])
    feature_cols.extend([c + "_Encoded" for c in categorical_cols if c + "_Encoded" in df_enc.columns])

    df_enc = df_enc.dropna(subset=feature_cols + ["Sales_Volume"]).copy()
    X = df_enc[feature_cols]
    y = df_enc["Sales_Volume"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=80, max_depth=10, random_state=42
        ),
    }

    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = mse_test**0.5
        mae_test = mean_absolute_error(y_test, y_pred_test)

        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring="r2")  # lighter
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append(
            {
                "Model": name,
                "Train R²": train_r2,
                "Test R²": test_r2,
                "Test RMSE": rmse_test,
                "Test MAE": mae_test,
                "CV R² (mean)": cv_mean,
                "CV R² (std)": cv_std,
            }
        )

    results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False)

    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = models_dict[best_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Feature importance via Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=120, max_depth=10, random_state=42
    )
    rf_model.fit(X_train, y_train)
    feature_importance = pd.DataFrame(
        {"Feature": feature_cols, "Importance": rf_model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    feature_importance["Feature_Name"] = feature_importance["Feature"].str.replace(
        "_Encoded", "", regex=False
    )

    return results_df, feature_importance, y_test, y_pred_best, best_name


def rq1_view(df_analysis: pd.DataFrame):
    st.markdown("### RQ1 – Key Factors Influencing Sales Performance")

    if len(df_analysis) < 50:
        st.warning("Not enough rows to reliably train ML models.")
        return

    run = st.button("Run / Refresh Models (cached)")
    if run or "rq1_ran" not in st.session_state:
        (
            results_df,
            feature_importance,
            y_test,
            y_pred_best,
            best_name,
        ) = train_rq1_models(df_analysis)
        st.session_state["rq1_results"] = (results_df, feature_importance, y_test, y_pred_best, best_name)
        st.session_state["rq1_ran"] = True

    if "rq1_results" not in st.session_state:
        st.info("Click the button above to train models for RQ1.")
        return

    results_df, feature_importance, y_test, y_pred_best, best_name = st.session_state["rq1_results"]

    st.markdown("#### Model Comparison")
    st.dataframe(
        results_df.style.format(
            {
                "Train R²": "{:.3f}",
                "Test R²": "{:.3f}",
                "Test RMSE": "{:,.0f}",
                "Test MAE": "{:,.0f}",
                "CV R² (mean)": "{:.3f}",
                "CV R² (std)": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    best_row = results_df.iloc[0]
    st.success(
        f"Best model: **{best_row['Model']}**  |  "
        f"Test R² = {best_row['Test R²']:.3f}, "
        f"RMSE ≈ {best_row['Test RMSE']:,.0f}"
    )

    # Feature importance
    st.markdown("#### Top Drivers of Sales Volume (Random Forest Importance)")
    top_feats = feature_importance.head(15)
    fig_imp = px.bar(
        top_feats.sort_values("Importance"),
        x="Importance",
        y="Feature_Name",
        orientation="h",
        title="Top 15 Features by Importance",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Predicted vs Actual
    st.markdown(f"#### Predicted vs Actual Sales Volume ({best_name})")
    fig_pred = px.scatter(
        x=y_test,
        y=y_pred_best,
        labels={"x": "Actual Sales Volume", "y": "Predicted Sales Volume"},
        title="Predicted vs Actual",
    )
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    fig_pred.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )
    )
    st.plotly_chart(fig_pred, use_container_width=True)


# -------------------------------------------------
# RQ2 – REVENUE CONTRIBUTION
# -------------------------------------------------
@st.cache_data
def rq2_aggregates(df_analysis: pd.DataFrame):
    revenue_by_region = (
        df_analysis.groupby("Region")
        .agg({"Revenue": "sum", "Sales_Volume": "sum"})
        .rename(columns={"Revenue": "Total_Revenue", "Sales_Volume": "Total_Sales"})
        .reset_index()
    )

    revenue_by_cat_region = (
        df_analysis.groupby(["Region", "Vehicle_Category"])["Revenue"]
        .sum()
        .reset_index()
    )

    yearly_revenue = (
        df_analysis.groupby("Year")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    return revenue_by_region, revenue_by_cat_region, yearly_revenue


def rq2_view(df_analysis: pd.DataFrame):
    st.markdown("### RQ2 – Revenue Contribution by Region & Category")

    revenue_by_region, revenue_by_cat_region, yearly_revenue = rq2_aggregates(df_analysis)

    view = st.selectbox(
        "Select view:",
        [
            "Revenue by Region",
            "Top Regions (Bar)",
            "Region x Vehicle Category (Heatmap)",
            "Yearly Revenue Trend",
        ],
    )

    if view == "Revenue by Region":
        fig = px.pie(
            revenue_by_region,
            values="Total_Revenue",
            names="Region",
            title="Revenue Distribution by Region",
            hole=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Table – Revenue by Region")
        tmp = revenue_by_region.copy()
        tmp["Total_Revenue_Millions"] = tmp["Total_Revenue"] / 1e6
        st.dataframe(
            tmp[["Region", "Total_Revenue_Millions", "Total_Sales"]].sort_values(
                "Total_Revenue_Millions", ascending=False
            ),
            use_container_width=True,
        )

    elif view == "Top Regions (Bar)":
        top_n = st.slider("Number of top regions", 3, len(revenue_by_region), 5)
        tmp = revenue_by_region.sort_values("Total_Revenue", ascending=False).head(top_n)
        tmp["Total_Revenue_Millions"] = tmp["Total_Revenue"] / 1e6

        fig = px.bar(
            tmp,
            x="Region",
            y="Total_Revenue_Millions",
            text_auto=".1f",
            title=f"Top {top_n} Regions by Revenue (Millions USD)",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Revenue (Millions USD)")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Region x Vehicle Category (Heatmap)":
        pivot = revenue_by_cat_region.pivot(
            index="Region", columns="Vehicle_Category", values="Revenue"
        ).fillna(0)
        fig = px.imshow(
            pivot / 1e6,
            text_auto=".1f",
            color_continuous_scale="YlGnBu",
            title="Revenue by Region & Vehicle Category (Millions USD)",
        )
        fig.update_xaxes(title="Vehicle Category")
        fig.update_yaxes(title="Region")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Yearly Revenue Trend":
        yearly_revenue["Revenue_Millions"] = yearly_revenue["Revenue"] / 1e6

        X_year = yearly_revenue["Year"].values
        y_rev = yearly_revenue["Revenue"].values
        slope, intercept, r_value, p_value_trend, std_err = linregress(X_year, y_rev)
        trend = intercept + slope * X_year

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yearly_revenue["Year"],
                y=yearly_revenue["Revenue_Millions"],
                mode="lines+markers",
                name="Actual Revenue",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=yearly_revenue["Year"],
                y=trend / 1e6,
                mode="lines",
                name="Trend Line",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Yearly Revenue Trend (Millions USD)",
            xaxis_title="Year",
            yaxis_title="Revenue (Millions USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Slope: **${slope:,.0f} / year**, R²: **{r_value**2:.3f}**, p-value: **{p_value_trend:.4f}**"
        )


# -------------------------------------------------
# RQ3 – CLUSTERING
# -------------------------------------------------
@st.cache_resource
def rq3_clusters(df_analysis: pd.DataFrame, k: int):
    cluster_df = df_analysis.copy()

    features = [
        "Price_USD",
        "Sales_Volume",
        "Mileage_KM",
        "Engine_Size_L",
        "Car_Age",
        "Performance_Score",
        "Revenue",
    ]
    cluster_df = cluster_df.dropna(subset=features).copy()
    X = cluster_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_df["Cluster"] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cluster_df["PCA1"] = X_pca[:, 0]
    cluster_df["PCA2"] = X_pca[:, 1]

    cluster_profiles = (
        cluster_df.groupby("Cluster")
        .agg(
            Avg_Price=("Price_USD", "mean"),
            Avg_Sales=("Sales_Volume", "mean"),
            Total_Revenue=("Revenue", "sum"),
            Avg_Mileage=("Mileage_KM", "mean"),
            Avg_Engine=("Engine_Size_L", "mean"),
            Count=("Model", "count"),
        )
        .round(2)
        .reset_index()
    )

    return cluster_df, cluster_profiles, sil_score, pca.explained_variance_ratio_


def rq3_view(df_analysis: pd.DataFrame):
    st.markdown("### RQ3 – Sales Behavior Patterns (Clustering)")

    if len(df_analysis) < 50:
        st.warning("Not enough rows to cluster meaningfully.")
        return

    k = st.slider("Number of clusters (k)", 2, 8, 4)
    cluster_df, cluster_profiles, sil_score, var_ratio = rq3_clusters(df_analysis, k)

    st.caption(f"Silhouette Score: **{sil_score:.3f}**  |  "
               f"PC1 var: **{var_ratio[0]*100:.1f}%**, PC2 var: **{var_ratio[1]*100:.1f}%**")

    st.markdown("#### Cluster Visualization (PCA)")
    fig_pca = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["Model", "Region", "Price_USD", "Sales_Volume"],
        title="Clusters in PCA Space",
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("#### Cluster Sizes")
    size_df = cluster_df["Cluster"].value_counts().reset_index()
    size_df.columns = ["Cluster", "Count"]
    fig_sizes = px.bar(
        size_df,
        x="Cluster",
        y="Count",
        text_auto=True,
        title="Cluster Sizes",
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

    st.markdown("#### Revenue by Cluster")
    rev_df = (
        cluster_df.groupby("Cluster")["Revenue"]
        .sum()
        .reset_index()
        .assign(Revenue_Millions=lambda d: d["Revenue"] / 1e6)
    )
    fig_rev = px.bar(
        rev_df,
        x="Cluster",
        y="Revenue_Millions",
        text_auto=".1f",
        title="Total Revenue by Cluster (Millions USD)",
    )
    fig_rev.update_layout(yaxis_title="Revenue (Millions USD)")
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("#### Cluster Profiles")
    st.dataframe(cluster_profiles, use_container_width=True)


# -------------------------------------------------
# RQ4 – FORECASTING
# -------------------------------------------------
@st.cache_data
def rq4_forecast(df_analysis: pd.DataFrame):
    yearly = (
        df_analysis.groupby("Year")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )
    yearly["Revenue_Millions"] = yearly["Revenue"] / 1e6

    X_year = yearly["Year"].values
    y_rev = yearly["Revenue"].values

    slope, intercept, r_value, p_value_trend, std_err = linregress(X_year, y_rev)
    trend = intercept + slope * X_year

    next_year = X_year.max() + 1
    forecast = intercept + slope * next_year

    residuals = y_rev - trend
    rmse = np.sqrt(np.mean(residuals**2))
    lower = (forecast - 1.96 * rmse) / 1e6
    upper = (forecast + 1.96 * rmse) / 1e6

    return yearly, trend, next_year, forecast, lower, upper, slope, r_value, p_value_trend


def rq4_view(df_analysis: pd.DataFrame):
    st.markdown("### RQ4 – Forecasting / Predictive Modeling")

    yearly, trend, next_year, forecast, lower, upper, slope, r_value, p_value_trend = rq4_forecast(df_analysis)

    st.markdown("#### Historical Revenue & Next-Year Forecast")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yearly["Year"],
            y=yearly["Revenue_Millions"],
            mode="lines+markers",
            name="Actual Revenue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["Year"],
            y=trend / 1e6,
            mode="lines",
            name="Trend Line",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_year],
            y=[forecast / 1e6],
            mode="markers",
            name=f"Forecast {next_year}",
            marker=dict(color="black", size=10, symbol="x"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_year, next_year],
            y=[lower, upper],
            mode="lines",
            name="95% CI",
            line=dict(color="black", dash="dot"),
        )
    )

    fig.update_layout(
        title="Revenue Trend & Next-Year Forecast (Millions USD)",
        xaxis_title="Year",
        yaxis_title="Revenue (Millions USD)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Next-year forecast ({next_year}): **${forecast/1e6:.1f}M** "
        f"(95% CI: **{lower:.1f}M – {upper:.1f}M**)\n\n"
        f"Slope: **${slope:,.0f} / year**, R²: **{r_value**2:.3f}**, p-value: **{p_value_trend:.4f}**"
    )

    st.markdown("#### Yearly Revenue & YoY Growth")
    yearly = yearly.copy()
    yearly["YoY_Growth_%"] = yearly["Revenue"].pct_change() * 100
    st.dataframe(
        yearly[["Year", "Revenue_Millions", "YoY_Growth_%"]].round(2),
        use_container_width=True,
    )


# -------------------------------------------------
# MAIN APP LAYOUT: TABS + DROPDOWNS
# -------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional – otherwise uses BMW Data Final Project.csv)",
    type=["csv"],
)

df = load_data(uploaded_file)
df_analysis = engineer_features(df)

st.sidebar.markdown("---")
st.sidebar.caption("📅 Analysis run on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

st.title("BMW Sales Analysis – Streamlined Streamlit Dashboard")

st.info(
    f"Dataset: **{len(df):,} rows**, Years: **{df['Year'].min()}–{df['Year'].max()}**, "
    f"Regions: **{df['Region'].nunique()}**, Models: **{df['Model'].nunique()}**"
)

tab_overview, tab_eda, tab_rq = st.tabs(
    ["Overview", "Exploratory Data Analysis (EDA)", "Research Questions"]
)

# --- Overview ---
with tab_overview:
    st.subheader("Section 2 – Data Loading & Initial Inspection")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### High-level stats")
        st.write(
            {
                "Total Records": len(df),
                "Year Range": f"{df['Year'].min()} - {df['Year'].max()}",
                "Regions": df["Region"].nunique(),
                "Models": df["Model"].nunique(),
                "Price Range": f"${df['Price_USD'].min():,.0f} - ${df['Price_USD'].max():,.0f}",
                "Sales Volume Range": f"{df['Sales_Volume'].min():,} - {df['Sales_Volume'].max():,}",
            }
        )

    with col2:
        st.markdown("#### df.info()")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.markdown("#### First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# --- EDA ---
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    eda_choice = st.selectbox(
        "Select EDA section:",
        [
            "3.1 – Distribution Analysis",
            "3.2 – Normality Tests",
            "3.3 – Relationship Analysis",
            "3.4 – Categorical Variable Analysis",
            "3.5 – Outlier Detection",
        ],
    )

    if eda_choice.startswith("3.1"):
        eda_distributions(df)
    elif eda_choice.startswith("3.2"):
        eda_normality(df)
    elif eda_choice.startswith("3.3"):
        eda_relationships(df)
    elif eda_choice.startswith("3.4"):
        eda_categorical(df)
    elif eda_choice.startswith("3.5"):
        eda_outliers(df)

# --- Research Questions ---
with tab_rq:
    st.subheader("Research Questions")

    rq_choice = st.selectbox(
        "Select research question:",
        [
            "RQ1 – Key Factors Influencing Sales",
            "RQ2 – Revenue Contribution by Region/Category",
            "RQ3 – Sales Behavior Patterns (Clustering)",
            "RQ4 – Forecasting / Predictive Modeling",
        ],
    )

    if rq_choice.startswith("RQ1"):
        rq1_view(df_analysis)
    elif rq_choice.startswith("RQ2"):
        rq2_view(df_analysis)
    elif rq_choice.startswith("RQ3"):
        rq3_view(df_analysis)
    elif rq_choice.startswith("RQ4"):
        rq4_view(df_analysis)

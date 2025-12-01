"""
BMW Sales Analysis - Comprehensive Research Dashboard
Team: Marcel Klibansky, Kevin Torano, Zachary Bramwell, Bernardo Sastre
A presentation-ready Streamlit dashboard for BMW sales analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import shapiro, skew, kurtosis, linregress, f_oneway

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="BMW Sales Analysis | Team Research Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066B1;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Info boxes */
    .insight-box {
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# BMW-inspired color palette
BMW_COLORS = {
    'primary': '#0066B1',      # BMW Blue
    'secondary': '#1C69D4',    # Lighter Blue
    'accent': '#E8E8E8',       # Silver
    'dark': '#1a1a2e',         # Dark navy
    'success': '#28a745',      # Green
    'warning': '#ffc107',      # Yellow
    'danger': '#dc3545',       # Red
}

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Set2

# =============================================================================
# DATA LOADING & FEATURE ENGINEERING
# =============================================================================
@st.cache_data
def load_data(uploaded_file=None):
    """Load the BMW sales dataset."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("BMW_Final_Data.csv")
    return df


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features for analysis."""
    df_analysis = df.copy()
    
    # Revenue calculation
    df_analysis["Revenue"] = df_analysis["Price_USD"]
    
    # Vehicle Category
    def categorize_vehicle(model):
        if model in ["X1", "X3", "X5", "X6"]:
            return "SUV"
        elif model in ["3 Series", "5 Series", "7 Series"]:
            return "Sedan"
        elif model in ["M3", "M5"]:
            return "Performance"
        elif model in ["i3", "i8"]:
            return "Electric/Hybrid"
        else:
            return "Other"
    
    df_analysis["Vehicle_Category"] = df_analysis["Model"].apply(categorize_vehicle)
    
    # Car Age
    current_year = int(df_analysis["Year"].max())
    df_analysis["Car_Age"] = current_year - df_analysis["Year"]
    
    # Price Category
    def categorize_price(p):
        if p < 50000:
            return "Budget"
        elif p < 80000:
            return "Mid-Range"
        else:
            return "Premium"
    
    df_analysis["Price_Category"] = df_analysis["Price_USD"].apply(categorize_price)
    
    # Mileage Category
    def categorize_mileage(m):
        if m < 50000:
            return "Low"
        elif m < 100000:
            return "Medium"
        elif m < 150000:
            return "High"
        else:
            return "Very High"
    
    df_analysis["Mileage_Category"] = df_analysis["Mileage_KM"].apply(categorize_mileage)
    
    # EV/Hybrid flag
    df_analysis["Is_EV_Hybrid"] = df_analysis["Fuel_Type"].isin(["Electric", "Hybrid"]).astype(int)
    
    # Performance Score
    df_analysis["Performance_Score"] = (
        df_analysis["Engine_Size_L"] * 100
        + (df_analysis["Transmission"] == "Automatic").astype(int) * 50
        + df_analysis["Is_EV_Hybrid"] * 75
    )
    
    # Price per Liter
    df_analysis["Price_per_Liter"] = df_analysis["Price_USD"] / df_analysis["Engine_Size_L"].replace(0, np.nan)
    
    # Estimated Depreciation Rate
    df_analysis["Estimated_Depreciation_Rate"] = df_analysis["Car_Age"] * 0.15
    
    return df_analysis


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_metric_card(value, label, prefix="", suffix=""):
    """Create a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def format_currency(value):
    """Format value as currency."""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.0f}"


# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
def show_introduction(df, df_analysis):
    """Display the introduction / title page."""
    
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <h1 style="color: #00d4ff; font-size: 3.5rem; font-weight: 800; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🚗 BMW GLOBAL SALES ANALYTICS
        </h1>
        <p style="color: #ffffff; font-size: 1.3rem; margin-top: 1rem; font-weight: 300; opacity: 0.95;">
            Advanced Data Intelligence | Strategic Market Insights | Predictive Analysis
        </p>
        <div style="margin-top: 1.5rem; padding: 0.8rem; background: rgba(0,212,255,0.1); border-left: 4px solid #00d4ff; border-radius: 8px;">
            <p style="color: #00d4ff; margin: 0; font-size: 1rem; font-weight: 500;">
                Comprehensive Research Study on BMW Vehicle Pricing, Sales Patterns & Market Dynamics
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Team info with enhanced design
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 8px 24px rgba(102,126,234,0.4);">
            <div style="display: inline-block; padding: 0.5rem 2rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: white; font-weight: 700; letter-spacing: 2px;">GROUP 4</h3>
            </div>
            <div style="height: 2px; width: 60%; margin: 1rem auto; background: linear-gradient(90deg, transparent, white, transparent);"></div>
            <p style="color: white; margin: 1rem 0 0 0; font-size: 1.1rem; font-weight: 400; line-height: 1.6;">
                Marcel Klibansky • Kevin Torano<br>Zachary Bramwell • Bernardo Sastre
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics Dashboard
    st.markdown("### 📊 Key Performance Indicators")
    
    total_revenue = df_analysis["Revenue"].sum()
    total_vehicles = len(df_analysis)
    avg_price = df_analysis["Price_USD"].mean()
    total_sales_volume = df_analysis["Sales_Volume"].sum()
    num_regions = df_analysis["Region"].nunique()
    num_models = df_analysis["Model"].nunique()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Revenue", format_currency(total_revenue))
    with col2:
        st.metric("Total Records", f"{total_vehicles:,}")
    with col3:
        st.metric("Avg Price", f"${avg_price:,.0f}")
    with col4:
        st.metric("Total Sales Volume", f"{total_sales_volume:,}")
    with col5:
        st.metric("Regions", num_regions)
    with col6:
        st.metric("Models", num_models)
    
    st.markdown("---")
    
    # Research Questions Overview
    st.markdown("### 🎯 Research Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: #1a1a2e; height: 150px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">RQ1: Key Factors Influencing Vehicle Pricing</h4>
            <p style="margin: 0; opacity: 1; font-size: 1.05rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">What are the key factors that most strongly determine BMW vehicle pricing across different models and regions?</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: #1a1a2e; height: 150px; display: flex; flex-direction: column; justify-content: center; margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">RQ2: Revenue Contribution Analysis</h4>
            <p style="margin: 0; opacity: 1; font-size: 1.05rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Which regions or vehicle categories contribute most to BMW's total sales revenue, and how has this changed over time?</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 12px; color: #1a1a2e; height: 150px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">RQ3: Sales Behavior Patterns</h4>
            <p style="margin: 0; opacity: 1; font-size: 1.05rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Can unsupervised learning reveal distinct sales behavior patterns across countries or models?</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 12px; color: #1a1a2e; height: 150px; display: flex; flex-direction: column; justify-content: center; margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">RQ4: Sales Forecasting</h4>
            <p style="margin: 0; opacity: 1; font-size: 1.05rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Can we build a predictive model to forecast BMW sales for the next year with confidence?</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology Overview
    st.markdown("### 🔬 Methodology Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **Data Preparation**
        - Data cleaning & validation
        - Feature engineering
        - Missing value analysis
        """)
    
    with col2:
        st.markdown("""
        **Exploratory Analysis**
        - Distribution analysis
        - Correlation studies
        - Outlier detection
        """)
    
    with col3:
        st.markdown("""
        **Machine Learning**
        - Regression models
        - Clustering (K-Means)
        - Cross-validation
        """)
    
    with col4:
        st.markdown("""
        **Statistical Testing**
        - Normality tests
        - ANOVA analysis
        - Trend significance
        """)


# =============================================================================
# SECTION 2: DATA OVERVIEW
# =============================================================================
def show_data_overview(df, df_analysis):
    """Display data loading and initial inspection."""
    
    st.markdown("## 📁 Data Loading & Initial Inspection")
    st.markdown("---")
    
    # Dataset Overview Cards
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(df.columns)}")
    with col3:
        st.metric("Time Period", f"{df['Year'].min()} - {df['Year'].max()}")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    with col5:
        st.metric("Duplicates", f"{df.duplicated().sum()}")
    
    st.markdown("---")
    
    # Data Structure
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Structure")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        st.markdown("### Column Types Summary")
        type_summary = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(type_summary, use_container_width=True)
    
    st.markdown("---")
    
    # Sample Data
    st.markdown("### Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().T.round(2), use_container_width=True)
    
    st.markdown("---")
    
    # Categorical Variables Summary
    st.markdown("### Categorical Variables Distribution")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(cat_cols):
        with col1 if i % 2 == 0 else col2:
            value_counts = df[col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
                labels={'x': col, 'y': 'Count'},
                color_discrete_sequence=[BMW_COLORS['primary']],
                text=value_counts.values
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=500,
                showlegend=False,
                xaxis_tickangle=-45,
                margin=dict(t=100, b=100, l=60, r=60),
                yaxis=dict(range=[0, value_counts.max() * 1.15])
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# =============================================================================
def show_eda(df, df_analysis):
    """Display comprehensive EDA."""
    
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    eda_section = st.selectbox(
        "Select Analysis Section:",
        [
            "3.1 - Distribution Analysis",
            "3.2 - Normality Tests",
            "3.3 - Correlation Analysis",
            "3.4 - Categorical Analysis",
            "3.5 - Outlier Detection"
        ]
    )
    
    if eda_section == "3.1 - Distribution Analysis":
        show_distribution_analysis(df)
    elif eda_section == "3.2 - Normality Tests":
        show_normality_tests(df)
    elif eda_section == "3.3 - Correlation Analysis":
        show_correlation_analysis(df, df_analysis)
    elif eda_section == "3.4 - Categorical Analysis":
        show_categorical_analysis(df)
    elif eda_section == "3.5 - Outlier Detection":
        show_outlier_detection(df)


def show_distribution_analysis(df):
    """Show distribution analysis for numeric variables."""
    
    st.markdown("### 3.1 Distribution Analysis")
    
    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    colors = [BMW_COLORS['primary'], BMW_COLORS['secondary'], '#28a745', '#ffc107']
    
    # Create individual bar chart distributions
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(numeric_cols):
        with col1 if i % 2 == 0 else col2:
            data = df[col].dropna()
            
            # Special handling for Engine_Size_L - use actual values instead of bins
            if col == "Engine_Size_L":
                value_counts = data.value_counts().sort_index()
                labels = [f"{val:.1f}L" for val in value_counts.index]
                counts = value_counts.values
            else:
                # Create bins for other continuous variables
                bins = pd.cut(data, bins=20)
                bin_counts = bins.value_counts().sort_index()
                labels = [f"{int(interval.left):,} - {int(interval.right):,}" for interval in bin_counts.index]
                counts = bin_counts.values
            
            fig = px.bar(
                x=labels,
                y=counts,
                title=f"{col.replace('_', ' ')} Distribution",
                labels={'x': col.replace('_', ' '), 'y': 'Count'},
                color_discrete_sequence=[colors[i]]
            )
            
            fig.update_traces(texttemplate='%{y:,}', textposition='outside')
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=450,
                showlegend=False,
                xaxis_tickangle=-45,
                xaxis_title=col.replace('_', ' ') + (" Size" if col == "Engine_Size_L" else " Range"),
                yaxis_title="Count",
                margin=dict(t=80, b=100),
                yaxis=dict(range=[0, counts.max() * 1.15])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate distribution statistics
            sk = skew(data)
            mean_val = data.mean()
            median_val = data.median()
            
            # Determine distribution type
            if abs(sk) < 0.5:
                dist_type = "Normal Distribution"
                dist_desc = "Data is symmetrically distributed around the mean"
            elif sk > 0:
                dist_type = "Right-Skewed Distribution"
                dist_desc = "Data has a longer tail on the right side (most values are lower)"
            else:
                dist_type = "Left-Skewed Distribution"
                dist_desc = "Data has a longer tail on the left side (most values are higher)"
            
            # Display interpretation box
            st.info(f"""
**{dist_type}**  
{dist_desc}  
*Skewness: {sk:.3f} | Mean: {mean_val:,.0f} | Median: {median_val:,.0f}*
            """)
    
    st.markdown("---")
    
    # Interpretation Guide
    st.info("""
    **Interpretation Guide:**
    - **Distribution Shape**: The bars show how frequently values occur in each range
    - **Higher Bars**: Indicate more common values in that range
    - **Spread**: Wide distribution indicates high variability; narrow indicates consistency
    - **Skewness**: If bars are taller on the left, the distribution is right-skewed (and vice versa)
    """)
    
    st.markdown("---")
    
    # Summary statistics table
    st.markdown("### Summary Statistics")
    
    stats_df = pd.DataFrame({
        'Variable': numeric_cols,
        'Mean': [df[col].mean() for col in numeric_cols],
        'Median': [df[col].median() for col in numeric_cols],
        'Std Dev': [df[col].std() for col in numeric_cols],
        'Min': [df[col].min() for col in numeric_cols],
        'Max': [df[col].max() for col in numeric_cols],
        'Skewness': [skew(df[col].dropna()) for col in numeric_cols],
        'Kurtosis': [kurtosis(df[col].dropna()) for col in numeric_cols]
    })
    
    st.dataframe(
        stats_df.style.format({
            'Mean': '{:,.2f}',
            'Median': '{:,.2f}',
            'Std Dev': '{:,.2f}',
            'Min': '{:,.2f}',
            'Max': '{:,.2f}',
            'Skewness': '{:.3f}',
            'Kurtosis': '{:.3f}'
        }),
        use_container_width=True
    )


def show_normality_tests(df):
    """Show normality test results."""
    
    st.markdown("### 3.2 Normality Tests")
    
    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        # Use sample if too large for Shapiro-Wilk
        sample_data = data.sample(min(5000, len(data)), random_state=42)
        stat, p_value = shapiro(sample_data)
        sk = skew(data)
        ku = kurtosis(data)
        
        # Determine shape based on skewness and kurtosis
        if abs(sk) < 0.5 and abs(ku) < 1:
            shape = "Approximately Normal"
        elif abs(sk) < 0.5:
            shape = "Symmetric (Heavy tails)" if ku > 1 else "Symmetric (Light tails)"
        else:
            shape = 'Right-skewed' if sk > 0 else 'Left-skewed'
        
        results.append({
            'Variable': col.replace('_', ' '),
            'Shapiro-Wilk Statistic': stat,
            'p-value': p_value,
            'Skewness': sk,
            'Kurtosis': ku,
            'Distribution Shape': shape
        })
    
    results_df = pd.DataFrame(results)
    
    st.dataframe(
        results_df.style.format({
            'Shapiro-Wilk Statistic': '{:.4f}',
            'p-value': '{:.4e}',
            'Skewness': '{:.3f}',
            'Kurtosis': '{:.3f}'
        }).apply(lambda x: ['background-color: #1a472a' if v < 0.05 else 'background-color: #4a1a1a' 
                            for v in results_df['p-value']], subset=['p-value'], axis=0),
        use_container_width=True
    )
    
    st.warning("""
    **⚠️ Note on Large Datasets:**  
    With 50,000 data points, the Shapiro-Wilk test is extremely sensitive and will detect even tiny deviations from perfect normality. 
    All p-values < 0.05 indicate statistical rejection of normality, but this is common with large samples.
    
    **For practical purposes**, look at **Skewness** and **Kurtosis**:
    - **Skewness near 0** (|skew| < 0.5): Distribution is symmetric, similar to normal
    - **Kurtosis near 0** (|kurt| < 1): Distribution has normal-like tail behavior
    """)
    
    st.info("""
    **Interpretation Guide:**
    - **Shapiro-Wilk p-value < 0.05**: Statistically rejects perfect normality (expected with large datasets)
    - **Skewness**: Measures asymmetry (0 = symmetric, positive = right tail, negative = left tail)
    - **Kurtosis**: Measures tail heaviness (0 = normal tails, positive = heavy tails, negative = light tails)
    - **Distribution Shape**: Based on skewness/kurtosis, describes the practical shape of the data
    """)
    
    # Q-Q Plots
    st.markdown("### Q-Q Plots")
    
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(numeric_cols[:2]):
        with col1 if i == 0 else col2:
            data = df[col].dropna().sample(min(1000, len(df)), random_state=42)
            sorted_data = np.sort(data)
            theoretical_quantiles = np.random.normal(data.mean(), data.std(), len(data))
            theoretical_quantiles = np.sort(theoretical_quantiles)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                marker=dict(color=BMW_COLORS['primary'], size=5, opacity=0.5),
                name='Data'
            ))
            fig.add_trace(go.Scatter(
                x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                y=[min(theoretical_quantiles), max(theoretical_quantiles)],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Normal Line'
            ))
            fig.update_layout(
                title=f"Q-Q Plot: {col.replace('_', ' ')}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template=PLOTLY_TEMPLATE,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation box for each Q-Q plot
            # Calculate how well data follows the line (rough assessment)
            from scipy.stats import pearsonr
            corr, _ = pearsonr(theoretical_quantiles, sorted_data)
            
            if corr > 0.99:
                interpretation = "Points closely follow the red line - data is approximately normally distributed"
            elif corr > 0.95:
                interpretation = "Points generally follow the red line with some deviation - data is reasonably normal"
            else:
                interpretation = "Points deviate significantly from the red line - data is NOT normally distributed"
            
            st.info(f"""
**Q-Q Plot Interpretation**  
{interpretation}  
*Correlation with normal line: {corr:.4f}*  
If points fall along the red dashed line, the data follows a normal distribution. Deviations indicate non-normality.
            """)
    
    st.markdown("---")
    
    # Overall Q-Q interpretation guide
    st.info("""
    **How to Read Q-Q Plots:**
    - **Points on the line**: Data is normally distributed
    - **S-shaped curve**: Distribution has heavier or lighter tails than normal
    - **Points above line on right**: Right-skewed distribution
    - **Points below line on left**: Left-skewed distribution
    """)


def show_correlation_analysis(df, df_analysis):
    """Show correlation analysis."""
    
    st.markdown("### 3.3 Correlation Analysis")
    
    # Correlation Matrix
    numeric_features = ["Price_USD", "Sales_Volume", "Engine_Size_L", "Mileage_KM", "Year"]
    corr_matrix = df[numeric_features].corr()
    
    # Display raw correlation values for debugging
    st.markdown("#### Correlation Matrix")
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.3f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Correlation Matrix - Numeric Variables",
        labels=dict(color="Correlation")
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation table
    st.markdown("#### Detailed Correlation Values")
    st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.4f}"), 
                 use_container_width=True)
    
    st.info("""
    **Interpretation Guide:**
    - **1.0**: Perfect positive correlation (variables move together)
    - **0.0**: No correlation (variables are independent)
    - **-1.0**: Perfect negative correlation (variables move opposite)
    - **|r| > 0.7**: Strong correlation
    - **|r| 0.3-0.7**: Moderate correlation
    - **|r| < 0.3**: Weak correlation
    """)
    
    st.warning("""
    **Note on Weak Correlations:**  
    The relatively weak correlations observed here are typical for automotive sales data. This indicates that:
    - **Complex relationships**: Sales are influenced by multiple factors beyond simple linear relationships
    - **Categorical importance**: Region, Model, and Fuel Type (categorical variables) often drive sales more than numeric variables alone
    - **Good for modeling**: Low correlation between predictors means less multicollinearity, which improves machine learning model performance
    - **Real-world behavior**: BMW sales patterns are multifaceted and cannot be explained by simple pairwise relationships
    
    Later sections (clustering, regression) will reveal more nuanced patterns that correlation analysis cannot capture.
    """)
    
    st.markdown("---")
    
    # Key correlations
    st.markdown("### Distribution Insights by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by Region
        fig = px.box(
            df,
            x="Region",
            y="Price_USD",
            color="Region",
            title="Price Distribution by Region",
            labels={"Price_USD": "Price (USD)", "Region": "Region"},
            color_discrete_sequence=COLOR_SEQUENCE
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show average prices
        avg_prices = df.groupby('Region')['Price_USD'].mean().sort_values(ascending=False)
        st.markdown("**Average Prices by Region:**")
        for region, price in avg_prices.items():
            st.write(f"• {region}: ${price:,.0f}")
    
    with col2:
        # Sales Volume distribution by Model
        top_models = df['Model'].value_counts().head(8).index
        df_top_models = df[df['Model'].isin(top_models)]
        
        fig = px.box(
            df_top_models,
            x="Model",
            y="Sales_Volume",
            color="Model",
            title="Sales Volume Distribution by Top Models",
            labels={"Sales_Volume": "Sales Volume", "Model": "Model"},
            color_discrete_sequence=COLOR_SEQUENCE
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show average sales
        avg_sales = df_top_models.groupby('Model')['Sales_Volume'].mean().sort_values(ascending=False)
        st.markdown("**Average Sales Volume:**")
        for model, sales in avg_sales.items():
            st.write(f"• {model}: {sales:,.0f}")


def show_categorical_analysis(df):
    """Show categorical variable analysis."""
    
    st.markdown("### 3.4 Categorical Variable Analysis")
    
    cat_var = st.selectbox(
        "Select Categorical Variable:",
        ["Model", "Region", "Fuel_Type", "Transmission", "Color", "Sales_Classification"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average Sales by Category
        grouped = df.groupby(cat_var)['Sales_Volume'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            grouped,
            x=cat_var,
            y='Sales_Volume',
            title=f"Average Sales Volume by {cat_var}",
            color='Sales_Volume',
            color_continuous_scale='Blues',
            text='Sales_Volume'
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=80, b=100, l=60, r=60),
            yaxis=dict(range=[0, grouped['Sales_Volume'].max() * 1.15])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average Price by Category
        grouped_price = df.groupby(cat_var)['Price_USD'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            grouped_price,
            x=cat_var,
            y='Price_USD',
            title=f"Average Price by {cat_var}",
            color='Price_USD',
            color_continuous_scale='Greens',
            text='Price_USD'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=80, b=100, l=60, r=60),
            yaxis=dict(range=[0, grouped_price['Price_USD'].max() * 1.15])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown(f"### {cat_var} Summary Statistics")
    
    summary = df.groupby(cat_var).agg({
        'Sales_Volume': ['mean', 'sum', 'std'],
        'Price_USD': ['mean', 'min', 'max'],
        'Model': 'count'
    }).round(2)
    summary.columns = ['Avg Sales', 'Total Sales', 'Sales Std', 'Avg Price', 'Min Price', 'Max Price', 'Count']
    summary = summary.sort_values('Total Sales', ascending=False)
    
    st.dataframe(summary, use_container_width=True)


def show_outlier_detection(df):
    """Show outlier detection analysis."""
    
    st.markdown("### 3.5 Outlier Detection")
    
    numeric_cols = ["Price_USD", "Sales_Volume", "Mileage_KM", "Engine_Size_L"]
    
    col = st.selectbox("Select Variable for Analysis:", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig = px.box(
            df,
            y=col,
            title=f"Box Plot: {col.replace('_', ' ')}",
            color_discrete_sequence=[BMW_COLORS['primary']]
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by region
        fig = px.box(
            df,
            x="Region",
            y=col,
            title=f"{col.replace('_', ' ')} by Region",
            color="Region",
            color_discrete_sequence=COLOR_SEQUENCE
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Outlier statistics
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_pct = len(outliers) / len(df) * 100
    
    st.markdown("### Outlier Statistics (IQR Method)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lower Bound", f"{lower_bound:,.2f}")
    with col2:
        st.metric("Upper Bound", f"{upper_bound:,.2f}")
    with col3:
        st.metric("Outlier Count", f"{len(outliers):,}")
    with col4:
        st.metric("Outlier %", f"{outlier_pct:.2f}%")


# =============================================================================
# SECTION 4: RESEARCH QUESTION 1
# =============================================================================
@st.cache_resource
def train_rq1_models(_df_analysis: pd.DataFrame):
    """Train models for RQ1 analysis - predicting vehicle pricing."""
    
    df_enc = _df_analysis.copy()
    
    # Remove any rows with missing or invalid price data
    df_enc = df_enc[df_enc['Price_USD'].notna()].copy()
    df_enc = df_enc[df_enc['Price_USD'] > 0].copy()
    
    # Encode categorical variables
    categorical_cols = ["Model", "Region", "Fuel_Type", "Transmission"]
    
    for col in categorical_cols:
        if col in df_enc.columns and df_enc[col].notna().any():
            le = LabelEncoder()
            df_enc[f"{col}_Encoded"] = le.fit_transform(df_enc[col].astype(str))
    
    # Use simple, reliable features
    feature_cols = []
    
    # Add numeric features if they exist and have variation
    numeric_features = ["Engine_Size_L", "Mileage_KM", "Year"]
    for feat in numeric_features:
        if feat in df_enc.columns and df_enc[feat].notna().any() and df_enc[feat].std() > 0:
            feature_cols.append(feat)
    
    # Add encoded categorical features
    encoded_features = ["Model_Encoded", "Region_Encoded", "Fuel_Type_Encoded", "Transmission_Encoded"]
    for feat in encoded_features:
        if feat in df_enc.columns:
            feature_cols.append(feat)
    
    if len(feature_cols) == 0:
        # Fallback: create dummy results
        results_df = pd.DataFrame([{
            'Model': 'No Valid Features',
            'Train R²': 0.0,
            'Test R²': 0.0,
            'CV R² Mean': 0.0,
            'CV R² Std': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Overfit Gap': 0.0
        }])
        feature_importance = pd.DataFrame({
            'Feature': ['No Features'],
            'Importance': [1.0],
            'Feature_Clean': ['No Features']
        })
        return results_df, feature_importance, np.array([0]), np.array([0]), 0.0, 1.0
    
    # Prepare data
    df_enc = df_enc.dropna(subset=feature_cols + ["Price_USD"]).copy()
    
    if len(df_enc) < 50:
        # Not enough data
        results_df = pd.DataFrame([{
            'Model': 'Insufficient Data',
            'Train R²': 0.0,
            'Test R²': 0.0,
            'CV R² Mean': 0.0,
            'CV R² Std': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Overfit Gap': 0.0
        }])
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': [1.0/len(feature_cols)] * len(feature_cols),
            'Feature_Clean': [f.replace('_Encoded', '').replace('_', ' ') for f in feature_cols]
        })
        return results_df, feature_importance, np.array([0]), np.array([0]), 0.0, 1.0
    
    X = df_enc[feature_cols].values
    y = df_enc["Price_USD"].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    models_dict = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }
    
    results = []
    
    for name, model in models_dict.items():
        try:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
            
            results.append({
                'Model': name,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'CV R² Mean': cv_scores.mean(),
                'CV R² Std': cv_scores.std(),
                'RMSE': rmse,
                'MAE': mae,
                'Overfit Gap': train_r2 - test_r2
            })
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    if len(results) == 0:
        # All models failed
        results_df = pd.DataFrame([{
            'Model': 'All Models Failed',
            'Train R²': 0.0,
            'Test R²': 0.0,
            'CV R² Mean': 0.0,
            'CV R² Std': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Overfit Gap': 0.0
        }])
    else:
        results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    # Feature importance
    try:
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        feature_importance['Feature_Clean'] = feature_importance['Feature'].str.replace('_Encoded', '').str.replace('_', ' ')
    except:
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': [1.0/len(feature_cols)] * len(feature_cols),
            'Feature_Clean': [f.replace('_Encoded', '').replace('_', ' ') for f in feature_cols]
        })
    
    # Get best model predictions
    if len(results) > 0:
        best_model_name = results_df.iloc[0]['Model']
        best_model = models_dict[best_model_name]
        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)
    else:
        y_pred_best = np.zeros_like(y_test)
    
    # ANOVA for regional differences - using Price instead of Sales
    try:
        regions = _df_analysis['Region'].unique()
        region_prices = [_df_analysis[_df_analysis['Region'] == r]['Price_USD'].dropna().values for r in regions]
        region_prices = [rp for rp in region_prices if len(rp) > 0]
        if len(region_prices) > 1:
            f_stat, p_value = f_oneway(*region_prices)
        else:
            f_stat, p_value = 0.0, 1.0
    except:
        f_stat, p_value = 0.0, 1.0
    
    return results_df, feature_importance, y_test, y_pred_best, f_stat, p_value
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    models_dict = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, min_samples_split=3, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }
    
    results = []
    
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
        
        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'RMSE': rmse,
            'MAE': mae,
            'Overfit Gap': train_r2 - test_r2
        })
    
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    # Feature importance from best tree-based model
    tree_models = {k: v for k, v in models_dict.items() if k in ["Random Forest", "Gradient Boosting", "Extra Trees"]}
    best_tree_model = None
    best_tree_r2 = -np.inf
    
    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_tree_r2:
            best_tree_r2 = r2
            best_tree_model = model
    
    if best_tree_model is not None and hasattr(best_tree_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_tree_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': [1/len(feature_cols)] * len(feature_cols)
        }).sort_values('Importance', ascending=False)
    
    feature_importance['Feature_Clean'] = feature_importance['Feature'].str.replace('_Encoded', '').str.replace('_', ' ')
    
    # Get best model predictions
    best_model_name = results_df.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
    # ANOVA for regional differences
    regions = _df_analysis['Region'].unique()
    region_sales = [_df_analysis[_df_analysis['Region'] == r]['Sales_Volume'].values for r in regions]
    f_stat, p_value = f_oneway(*region_sales)
    
    return results_df, feature_importance, y_test.values, y_pred_best, f_stat, p_value


def show_rq1(df_analysis):
    """Display Research Question 1 analysis."""
    
    st.markdown("## 🔬 RQ1: Key Factors Influencing Vehicle Pricing")
    st.markdown("""
    > **Question:** What are the key factors that most strongly determine BMW vehicle pricing across different models and regions?
    """)
    st.markdown("---")
    
    # Price Analysis by Model and Region
    st.markdown("### Pricing Patterns Across Models and Regions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average price by model
        model_price = df_analysis.groupby('Model')['Price_USD'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).reset_index()
        
        fig = px.bar(
            model_price,
            x='Model',
            y='mean',
            title="Average Vehicle Price by Model",
            labels={'mean': 'Average Price (USD)', 'Model': 'BMW Model'},
            color='mean',
            color_continuous_scale='Blues',
            text='mean'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=450,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(t=80, b=120, l=60, r=60),
            yaxis=dict(range=[0, model_price['mean'].max() * 1.15])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model price insights
        highest_model = model_price.iloc[0]
        lowest_model = model_price.iloc[-1]
        price_range = highest_model['mean'] - lowest_model['mean']
        
        st.info(f"""
**Model Pricing Insights:**  
💰 **Highest**: {highest_model['Model']} at ${highest_model['mean']:,.0f} (±${highest_model['std']:,.0f})  
💵 **Lowest**: {lowest_model['Model']} at  ${lowest_model['mean']:,.0f} (±${lowest_model['std']:,.0f})  
📊 **Price Range**: ${price_range:,.0f} difference between highest and lowest models  
        """)
    
    with col2:
        # Average price by region
        region_price = df_analysis.groupby('Region')['Price_USD'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).reset_index()
        
        fig = px.bar(
            region_price,
            x='Region',
            y='mean',
            title="Average Vehicle Price by Region",
            labels={'mean': 'Average Price (USD)', 'Region': 'Region'},
            color='mean',
            color_continuous_scale='Reds',
            text='mean'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=450,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(t=80, b=120, l=60, r=60),
            yaxis=dict(range=[0, region_price['mean'].max() * 1.15])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional price insights
        highest_region = region_price.iloc[0]
        lowest_region = region_price.iloc[-1]
        region_range = highest_region['mean'] - lowest_region['mean']
        
        st.info(f"""
**Regional Pricing Insights:**  
🌍 **Highest**: {highest_region['Region']} at ${highest_region['mean']:,.0f}  
🌎 **Lowest**: {lowest_region['Region']} at ${lowest_region['mean']:,.0f}  
📊 **Regional Variation**: ${region_range:,.0f} price difference across regions  
        """)
    
    # Heatmap: Average price by Model and Region
    st.markdown("### Price Heatmap: Model × Region")
    
    pivot_data = df_analysis.pivot_table(
        values='Price_USD',
        index='Model',
        columns='Region',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Viridis',
        text=pivot_data.values.round(0),
        texttemplate='$%{text:,.0f}',
        textfont={"size": 10},
        colorbar=dict(title="Avg Price (USD)")
    ))
    
    fig.update_layout(
        title="Average BMW Pricing Across Models and Regions",
        xaxis_title="Region",
        yaxis_title="Model",
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=100, r=100, t=80, b=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate specific insights from the heatmap data
    # Find highest and lowest priced model-region combinations
    pivot_flat = pivot_data.stack().reset_index()
    pivot_flat.columns = ['Model', 'Region', 'Price']
    pivot_flat = pivot_flat.sort_values('Price', ascending=False)
    
    highest_combo = pivot_flat.iloc[0]
    lowest_combo = pivot_flat.iloc[-1]
    
    # Calculate regional premium (max region vs min region)
    regional_avg = pivot_data.mean(axis=0).sort_values(ascending=False)
    highest_region_avg = regional_avg.iloc[0]
    lowest_region_avg = regional_avg.iloc[-1]
    regional_premium_pct = ((highest_region_avg - lowest_region_avg) / lowest_region_avg) * 100
    
    # Calculate model premium (max model vs min model)
    model_avg = pivot_data.mean(axis=1).sort_values(ascending=False)
    highest_model_avg = model_avg.iloc[0]
    lowest_model_avg = model_avg.iloc[-1]
    model_price_range = highest_model_avg - lowest_model_avg
    
    # Calculate regional premium (max region vs min region)
    regional_avg = pivot_data.mean(axis=0).sort_values(ascending=False)
    highest_region_avg = regional_avg.iloc[0]
    lowest_region_avg = regional_avg.iloc[-1]
    regional_price_range = highest_region_avg - lowest_region_avg
    
    st.markdown("---")
    
    # Machine Learning Analysis
    st.markdown("### Machine Learning: Quantifying Price Determinants")
    
    with st.spinner("Training models... This may take a moment."):
        results_df, feature_importance, y_test, y_pred, f_stat, p_value = train_rq1_models(df_analysis)
    
    # NOW show key findings with feature_importance available
    # Filter out 'Model' from features since the question asks what determines price ACROSS models
    feature_importance_filtered = feature_importance[feature_importance['Feature_Clean'] != 'Model'].reset_index(drop=True)
    
    st.success(f"""
**📊 Key Findings - Direct Answer to Research Question:**

**ANSWER: The key factors determining BMW pricing across different models and regions are:**

**1. {feature_importance_filtered.iloc[0]['Feature_Clean'].upper()}**  
Importance: {feature_importance_filtered.iloc[0]['Importance']*100:.1f} % of explained variance  
• The most significant vehicle characteristic affecting price  
• Newer vehicles command premium pricing across all model lines

**2. {feature_importance_filtered.iloc[1]['Feature_Clean'].upper()}**  
Importance: {feature_importance_filtered.iloc[1]['Importance']*100:.1f} % of explained variance  
• Second most influential pricing determinant  
• Higher mileage results in depreciation across all BMW models

**3. {feature_importance_filtered.iloc[2]['Feature_Clean'].upper()}**  
Importance: {feature_importance_filtered.iloc[2]['Importance']*100:.1f} % of explained variance  
• Engine specifications contribute to vehicle valuation  
• Larger engines typically associated with higher-end models

**How These Factors Vary Across Models:**  
🚗 Price range across model types: $ {model_price_range:,.0f}

• Premium models: {model_avg.index[0]} averages $ {highest_model_avg:,.0f}

• Entry models: {model_avg.index[-1]} averages $ {lowest_model_avg:,.0f}

**How These Factors Vary Across Regions:**  
🌍 Geographic pricing variation: $ {regional_price_range:,.0f}

• Premium market: {regional_avg.index[0]} averages $ {highest_region_avg:,.0f}

• Value market: {regional_avg.index[-1]} averages $ {lowest_region_avg:,.0f}

**Extreme Combinations:**  
💰 Highest: {highest_combo['Model']} in {highest_combo['Region']} at $ {highest_combo['Price']:,.0f}

💵 Lowest: {lowest_combo['Model']} in {lowest_combo['Region']} at $ {lowest_combo['Price']:,.0f}
    """)
    
    st.markdown("---")
    
    # Model Performance Comparison
    st.markdown("### Model Performance Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Train R²',
            x=results_df['Model'],
            y=results_df['Train R²'],
            marker_color=BMW_COLORS['primary'],
            text=results_df['Train R²'].round(4),
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Test R²',
            x=results_df['Model'],
            y=results_df['Test R²'],
            marker_color=BMW_COLORS['secondary'],
            text=results_df['Test R²'].round(4),
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='CV R² Mean',
            x=results_df['Model'],
            y=results_df['CV R² Mean'],
            marker_color='#28a745',
            text=results_df['CV R² Mean'].round(4),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="R² Score",
            barmode='group',
            template=PLOTLY_TEMPLATE,
            height=450,
            margin=dict(t=80, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        best = results_df.iloc[0]
        st.markdown("### Best Model")
        st.success(f"**{best['Model']}**")
        st.metric("Test R²", f"{best['Test R²']:.4f}")
        st.metric("RMSE", f"${best['RMSE']:,.0f}")
        st.metric("MAE", f"${best['MAE']:,.0f}")
    
    # Results table
    st.markdown("### Detailed Model Results")
    st.dataframe(
        results_df.style.format({
            'Train R²': '{:.4f}',
            'Test R²': '{:.4f}',
            'CV R² Mean': '{:.4f}',
            'CV R² Std': '{:.4f}',
            'RMSE': '${:,.0f}',
            'MAE': '${:,.0f}',
            'Overfit Gap': '{:.4f}'
        }).background_gradient(subset=['Test R²'], cmap='Greens'),
        use_container_width=True
    )
    
    # Model interpretation
    best_model = results_df.iloc[0]
    
    r2_interpretation = f"The model explains {best_model['Test R²']*100:.1f}% of price variation"
    overfit_status = 'minimal overfitting - model generalizes well' if best_model['Overfit Gap'] < 0.1 else 'some overfitting - consider regularization'
    
    st.info(f"""
**Model Performance Interpretation:**  
✅ **Best Model**: {best_model['Model']} with Test R² of {best_model['Test R²']:.4f}  
📊 **Explained Variance**: {r2_interpretation}  
🎯 **Prediction Accuracy**: Average error (MAE) of ${best_model['MAE']:,.0f}  
⚖️ **Overfitting Check**: Train-Test gap of {best_model['Overfit Gap']:.4f} indicates {overfit_status}  
    """)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### Feature Importance Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_features = feature_importance.head(15)
        
        fig = px.bar(
            top_features.sort_values('Importance'),
            x='Importance',
            y='Feature_Clean',
            orientation='h',
            title="Top 15 Features by Importance (Random Forest)",
            color='Importance',
            color_continuous_scale='Blues',
            text='Importance'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=550,
            yaxis_title="",
            xaxis_title="Importance Score",
            showlegend=False,
            margin=dict(l=150, r=80)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top 5 Drivers")
        for i, row in feature_importance.head(5).iterrows():
            pct = row['Importance'] * 100
            st.markdown(f"**{i+1}. {row['Feature_Clean']}**")
            st.progress(min(row['Importance'] * 2, 1.0))
            st.caption(f"{pct:.1f}% importance")
        
        # Cumulative importance
        cumsum = feature_importance['Importance'].cumsum()
        features_80 = (cumsum <= 0.80).sum() + 1
        st.info(f"📊 **{features_80} features** explain 80% of the variance")
    
    # Feature importance interpretation
    top_feature = feature_importance.iloc[0]
    st.success(f"""
**Key Drivers of Vehicle Pricing:**  
🥇 **Most Important Factor**: {top_feature['Feature_Clean']} ({top_feature['Importance']*100:.1f}% importance)  
📈 **Top 3 Combined Impact**: The top 3 features account for {feature_importance.head(3)['Importance'].sum()*100:.1f}% of price determination  
💡 **Actionable Insight**: Focus on the top {features_80} features when setting pricing strategy  
🎯 **Strategic Recommendation**: {top_feature['Feature_Clean']} is the primary price determinant and should guide pricing decisions  
    """)
    
    st.markdown("---")
    
    # Predicted vs Actual
    st.markdown("### Model Validation: Predicted vs Actual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(color=BMW_COLORS['primary'], opacity=0.5, size=5),
            name='Predictions'
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        ))
        
        fig.update_layout(
            title="Predicted vs Actual Price",
            xaxis_title="Actual Price (USD)",
            yaxis_title="Predicted Price (USD)",
            template=PLOTLY_TEMPLATE,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals distribution
        residuals = y_test - y_pred
        
        fig = px.histogram(
            x=residuals,
            nbins=50,
            title="Residuals Distribution",
            labels={'x': 'Residual (USD)', 'y': 'Frequency'},
            color_discrete_sequence=[BMW_COLORS['secondary']]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model validation interpretation
    from scipy.stats import pearsonr
    pred_corr, _ = pearsonr(y_test, y_pred)
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    
    st.info(f"""
**Model Validation Results:**  
📊 **Prediction Correlation**: {pred_corr:.4f} - {'Excellent' if pred_corr > 0.9 else 'Good' if pred_corr > 0.7 else 'Moderate'} alignment between actual and predicted values  
📉 **Residual Analysis**: Mean residual of ${residual_mean:,.0f} (close to $0 is ideal)  
🎯 **Prediction Consistency**: Standard deviation of ${residual_std:,.0f} indicates prediction variability  
✅ **Model Reliability**: {'Model shows strong predictive power and can be trusted for forecasting' if pred_corr > 0.8 else 'Model has moderate predictive power - use with caution'}  
    """)
    
    st.markdown("---")
    
    # Regional ANOVA
    st.markdown("### Regional Differences (ANOVA)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F-Statistic", f"{f_stat:.4f}")
    with col2:
        st.metric("p-value", f"{p_value:.6f}")
    with col3:
        significant = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
        st.metric("Result (α=0.05)", significant)
    
    if p_value < 0.05:
        st.success("**Conclusion:** There are statistically significant differences in vehicle pricing across regions.")
        st.info("""
**Regional Pricing Insights:**  
🌍 **Regional Variation Confirmed**: Statistical evidence shows BMW pricing differs significantly by region  
📊 **Strategic Implication**: Region-specific pricing strategies are necessary based on local market conditions  
🎯 **Recommendation**: Analyze regional price premiums to optimize pricing strategy in each market  
💡 **Action Items**: Consider local economic conditions, competition, and willingness to pay when setting regional prices  
        """)
    else:
        st.warning("**Conclusion:** No statistically significant differences in vehicle pricing across regions.")
        st.info("""
**Regional Pricing Insights:**  
🌍 **Regional Consistency**: Pricing is relatively uniform across regions  
📊 **Strategic Implication**: Standardized global pricing strategy is appropriate  
🎯 **Recommendation**: Focus on product features and model positioning rather than regional price variation  
💡 **Action Items**: Maintain consistent brand positioning and pricing globally  
        """)


# =============================================================================
# SECTION 5: RESEARCH QUESTION 2
# =============================================================================
def show_rq2(df_analysis):
    """Display Research Question 2 analysis."""
    
    st.markdown("## 💰 RQ2: Revenue Contribution Analysis")
    st.markdown("""
    > **Question:** Which regions or vehicle categories contribute most to BMW's total sales revenue, and how has this changed over time?
    """)
    st.markdown("---")
    
    # Calculate aggregates
    revenue_by_region = df_analysis.groupby('Region').agg({
        'Revenue': 'sum',
        'Sales_Volume': 'sum',
        'Price_USD': 'mean',
        'Model': 'count'
    }).reset_index()
    revenue_by_region.columns = ['Region', 'Total_Revenue', 'Total_Sales', 'Avg_Price', 'Vehicles_Sold']
    revenue_by_region['Revenue_Share'] = revenue_by_region['Total_Revenue'] / revenue_by_region['Total_Revenue'].sum() * 100
    revenue_by_region = revenue_by_region.sort_values('Total_Revenue', ascending=False)
    
    revenue_by_category = df_analysis.groupby('Vehicle_Category').agg({
        'Revenue': 'sum',
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()
    revenue_by_category.columns = ['Category', 'Total_Revenue', 'Total_Sales', 'Avg_Price']
    revenue_by_category['Revenue_Share'] = revenue_by_category['Total_Revenue'] / revenue_by_category['Total_Revenue'].sum() * 100
    revenue_by_category = revenue_by_category.sort_values('Total_Revenue', ascending=False)
    
    yearly_revenue = df_analysis.groupby('Year')['Revenue'].sum().reset_index()
    yearly_revenue['Revenue_Millions'] = yearly_revenue['Revenue'] / 1e6
    yearly_revenue['YoY_Growth'] = yearly_revenue['Revenue'].pct_change() * 100
    
    # Revenue by Region
    st.markdown("### Revenue Distribution by Region")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            revenue_by_region,
            values='Total_Revenue',
            names='Region',
            title="Revenue Share by Region",
            hole=0.4,
            color_discrete_sequence=COLOR_SEQUENCE
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            revenue_by_region,
            x='Region',
            y='Total_Revenue',
            title="Total Revenue by Region",
            color='Total_Revenue',
            color_continuous_scale='Blues',
            text=revenue_by_region['Total_Revenue'].apply(lambda x: f'${x/1e6:.0f}M')
        )
        fig.update_traces(textposition='inside', textfont_size=11)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(t=60, b=100, l=60, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Region table
    st.markdown("### Regional Performance Summary")
    display_df = revenue_by_region.copy()
    display_df['Total_Revenue'] = display_df['Total_Revenue'].apply(lambda x: f'${x/1e6:.1f}M')
    display_df['Total_Sales'] = display_df['Total_Sales'].apply(lambda x: f'{x:,.0f}')
    display_df['Avg_Price'] = display_df['Avg_Price'].apply(lambda x: f'${x:,.0f}')
    display_df['Revenue_Share'] = display_df['Revenue_Share'].apply(lambda x: f'{x:.1f}%')
    st.dataframe(display_df, use_container_width=True)
    
    # Regional insights box
    top_region = revenue_by_region.iloc[0]
    bottom_region = revenue_by_region.iloc[-1]
    region_range = top_region['Total_Revenue'] - bottom_region['Total_Revenue']
    
    st.info(f"""
**Regional Revenue Insights:**

• Top Market: {top_region['Region']} generates ${top_region['Total_Revenue']/1e6:.1f}M ({top_region['Revenue_Share']:.1f}% of total revenue)

• Value Market: {bottom_region['Region']} contributes ${bottom_region['Total_Revenue']/1e6:.1f}M ({bottom_region['Revenue_Share']:.1f}% of total revenue)

• Revenue Range: ${region_range/1e6:.1f}M difference between highest and lowest markets

• Market Concentration: Top 3 regions account for {revenue_by_region.head(3)['Revenue_Share'].sum():.1f}% of global revenue
    """)
    
    st.markdown("---")
    
    # Revenue by Vehicle Category
    st.markdown("### Revenue Distribution by Vehicle Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            revenue_by_category,
            values='Total_Revenue',
            names='Category',
            title="Revenue Share by Vehicle Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            revenue_by_category,
            x='Category',
            y='Total_Revenue',
            title="Total Revenue by Category",
            color='Total_Revenue',
            color_continuous_scale='Greens',
            text=revenue_by_category['Total_Revenue'].apply(lambda x: f'${x/1e6:.0f}M')
        )
        fig.update_traces(textposition='inside', textfont_size=11)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(t=60, b=100, l=60, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category insights box
    top_category = revenue_by_category.iloc[0]
    bottom_category = revenue_by_category.iloc[-1]
    
    st.info(f"""
**Vehicle Category Revenue Insights:**

• Leading Category: {top_category['Category']} dominates with ${top_category['Total_Revenue']/1e6:.1f}M ({top_category['Revenue_Share']:.1f}% share)

• Lowest Category: {bottom_category['Category']} generates ${bottom_category['Total_Revenue']/1e6:.1f}M ({bottom_category['Revenue_Share']:.1f}% share)

• Average Price Premium: {top_category['Category']} commands ${top_category['Avg_Price']:,.0f} average price

• Category Diversity: Revenue distributed across {len(revenue_by_category)} distinct vehicle categories
    """)
    
    st.markdown("---")
    
    # Yearly Trend Analysis
    st.markdown("### Revenue Trend Over Time")
    
    # Calculate trend line
    X_year = yearly_revenue['Year'].values
    y_rev = yearly_revenue['Revenue'].values
    slope, intercept, r_value, p_value, std_err = linregress(X_year, y_rev)
    trend_line = intercept + slope * X_year
    
    # Calculate CAGR
    first_year_rev = yearly_revenue.iloc[0]['Revenue']
    last_year_rev = yearly_revenue.iloc[-1]['Revenue']
    first_year = yearly_revenue.iloc[0]['Year']
    last_year = yearly_revenue.iloc[-1]['Year']
    n_years = last_year - first_year
    cagr = ((last_year_rev / first_year_rev) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_revenue['Year'],
        y=yearly_revenue['Revenue_Millions'],
        mode='lines+markers',
        name='Actual Revenue',
        line=dict(color=BMW_COLORS['primary'], width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_revenue['Year'],
        y=trend_line / 1e6,
        mode='lines',
        name=f'Trend Line (R²={r_value**2:.3f})',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Annual Revenue Trend (2010-2024)",
        xaxis_title="Year",
        yaxis_title="Revenue (Millions USD)",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CAGR", f"{cagr:.2f}%")
    with col2:
        st.metric("Trend Slope", f"${slope/1e6:.2f}M/year")
    with col3:
        st.metric("R² Value", f"{r_value**2:.4f}")
    with col4:
        trend_sig = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
        st.metric("Trend Significance", trend_sig)
    
    st.markdown("---")
    
    # Year-over-Year Growth
    st.markdown("### Year-over-Year Revenue Growth")
    
    fig = go.Figure()
    
    colors = ['green' if x >= 0 else 'red' for x in yearly_revenue['YoY_Growth'].fillna(0)]
    
    fig.add_trace(go.Bar(
        x=yearly_revenue['Year'],
        y=yearly_revenue['YoY_Growth'],
        marker_color=colors,
        text=yearly_revenue['YoY_Growth'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else ''),
        textposition='inside',
        textfont=dict(size=10, color='white')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title="Year-over-Year Revenue Growth Rate",
        xaxis_title="Year",
        yaxis_title="Growth Rate (%)",
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(t=60, b=60, l=60, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Regional Evolution
    st.markdown("### Regional Revenue Evolution Over Time")
    
    regional_yearly = df_analysis.groupby(['Year', 'Region'])['Revenue'].sum().reset_index()
    
    fig = px.line(
        regional_yearly,
        x='Year',
        y='Revenue',
        color='Region',
        title="Revenue Evolution by Region",
        labels={'Revenue': 'Revenue (USD)'},
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.markdown("### Revenue Heatmap: Region × Vehicle Category")
    
    pivot = df_analysis.pivot_table(
        values='Revenue',
        index='Region',
        columns='Vehicle_Category',
        aggfunc='sum'
    ).fillna(0)
    
    fig = px.imshow(
        pivot / 1e6,
        text_auto='.1f',
        color_continuous_scale='YlGnBu',
        title="Revenue by Region and Vehicle Category (Millions USD)",
        labels=dict(color="Revenue ($M)")
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        margin=dict(t=80, b=100, l=150, r=100),
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap insights
    max_combo = pivot.max().max()
    max_region_cat = pivot.stack().idxmax()
    
    st.info(f"""
**Regional × Category Insights:**

• Strongest Combination: {max_region_cat[0]} × {max_region_cat[1]} generates ${max_combo/1e6:.1f}M in revenue

• Regional Specialization: Different continents show distinct category preferences (visible in heatmap intensity patterns)

• Cross-Market Opportunities: Lighter areas indicate potential for category expansion in specific regions
    """)
    
    st.markdown("---")
    
    # Answer to Research Question
    st.success(f"""
**📊 Answer to Research Question:**

**Which regions and vehicle categories contribute most to BMW's revenue?**

**BY REGION:**

{chr(10).join([f"• {row['Region']}: ${row['Total_Revenue']/1e6:.1f}M ({row['Revenue_Share']:.1f}% of total)" for _, row in revenue_by_region.head(3).iterrows()])}

**BY VEHICLE CATEGORY:**

{chr(10).join([f"• {row['Category']}: ${row['Total_Revenue']/1e6:.1f}M ({row['Revenue_Share']:.1f}% of total)" for _, row in revenue_by_category.head(3).iterrows()])}

**REVENUE TRENDS OVER TIME:**

• CAGR: {cagr:.2f}% annual compound growth rate from {first_year} to {last_year}

• Trend Significance: Statistical analysis confirms {trend_sig.lower()} revenue trend (p-value < 0.05)

• Growth Pattern: Relatively stable year-over-year growth with {(yearly_revenue['YoY_Growth'] > 0).sum() - 1} years showing positive growth

**KEY FINDINGS:**

• Revenue is relatively balanced across major markets, with no single region dominating (top region < 25% share)

• {revenue_by_category.iloc[0]['Category']} category leads revenue generation, indicating strong demand for this segment

• Consistent upward revenue trend demonstrates BMW's sustained market strength across the 15-year period

• Regional evolution shows steady growth across all continents, with no major market disruptions
    """)


# =============================================================================
# SECTION 6: RESEARCH QUESTION 3
# =============================================================================
@st.cache_resource
def perform_clustering(_df_analysis: pd.DataFrame, k: int):
    """Perform K-Means clustering."""
    
    features = ['Price_USD', 'Sales_Volume', 'Mileage_KM', 'Engine_Size_L', 'Car_Age', 'Performance_Score']
    
    df_cluster = _df_analysis.dropna(subset=features).copy()
    X = df_cluster[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow method data
    inertias = []
    silhouettes = []
    K_range = range(2, 11)
    
    for k_val in K_range:
        kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans_temp.labels_))
    
    # Final clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_cluster['Cluster'] = cluster_labels
    
    # Metrics
    sil_score = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_cluster['PCA1'] = X_pca[:, 0]
    df_cluster['PCA2'] = X_pca[:, 1]
    
    # Cluster profiles
    cluster_profiles = df_cluster.groupby('Cluster').agg({
        'Price_USD': 'mean',
        'Sales_Volume': 'mean',
        'Mileage_KM': 'mean',
        'Engine_Size_L': 'mean',
        'Revenue': ['sum', 'mean'],
        'Model': 'count',
        'Is_EV_Hybrid': 'mean'
    }).round(2)
    cluster_profiles.columns = ['Avg_Price', 'Avg_Sales', 'Avg_Mileage', 'Avg_Engine', 
                                 'Total_Revenue', 'Avg_Revenue', 'Count', 'EV_Hybrid_Ratio']
    
    return df_cluster, cluster_profiles, sil_score, ch_score, db_score, list(K_range), inertias, silhouettes, pca.explained_variance_ratio_


def show_rq3(df_analysis):
    """Display Research Question 3 analysis."""
    
    st.markdown("## 🎯 RQ3: Sales Behavior Patterns (Clustering)")
    st.markdown("""
    > **Question:** Can unsupervised learning reveal distinct sales behavior patterns across continents or models?
    """)
    st.markdown("---")
    
    # First, run clustering for all K values to find optimal
    with st.spinner("Finding optimal number of clusters..."):
        # Quick analysis to find optimal K
        X_temp = df_analysis[['Year', 'Mileage_KM', 'Engine_Size_L', 'Price_USD', 'Sales_Volume']].copy()
        scaler_temp = StandardScaler()
        X_scaled_temp = scaler_temp.fit_transform(X_temp)
        
        K_range_test = range(2, 9)
        silhouettes_test = []
        
        for k_val in K_range_test:
            kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled_temp)
            silhouettes_test.append(silhouette_score(X_scaled_temp, labels_temp))
        
        optimal_k = K_range_test[silhouettes_test.index(max(silhouettes_test))]
    
    st.info(f"""
    **📊 Optimal Cluster Selection:** Based on silhouette score analysis, the optimal number of clusters is **K = {optimal_k}** 
    (Silhouette Score: {max(silhouettes_test):.4f})
    
    You can adjust the slider below to explore different cluster configurations.
    """)
    
    # Cluster selection with optimal K as default
    k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=optimal_k)
    
    with st.spinner("Performing clustering analysis..."):
        df_cluster, cluster_profiles, sil_score, ch_score, db_score, K_range, inertias, silhouettes, var_ratio = perform_clustering(df_analysis, k)
    
    # Elbow Method & Silhouette Score
    st.markdown("### Optimal K Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(K_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color=BMW_COLORS['primary'], width=3),
            marker=dict(size=10)
        ))
        fig.add_vline(x=k, line_dash="dash", line_color="red", annotation_text=f"Selected K={k}")
        fig.update_layout(
            title="Elbow Method",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia",
            template=PLOTLY_TEMPLATE,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(K_range),
            y=silhouettes,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#28a745', width=3),
            marker=dict(size=10)
        ))
        fig.add_vline(x=k, line_dash="dash", line_color="red", annotation_text=f"Selected K={k}")
        fig.update_layout(
            title="Silhouette Score by K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score",
            template=PLOTLY_TEMPLATE,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering Quality Metrics
    st.markdown("### Clustering Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Silhouette Score", f"{sil_score:.4f}")
        interpretation = "Good" if sil_score > 0.5 else ("Fair" if sil_score > 0.25 else "Weak")
        st.caption(f"Interpretation: {interpretation}")
    with col2:
        st.metric("Calinski-Harabasz", f"{ch_score:.0f}")
        st.caption("Higher is better")
    with col3:
        st.metric("Davies-Bouldin", f"{db_score:.4f}")
        st.caption("Lower is better")
    with col4:
        st.metric("PCA Variance Explained", f"{sum(var_ratio)*100:.1f}%")
        st.caption(f"PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%")
    
    st.markdown("---")
    
    # Cluster Visualization
    st.markdown("### Cluster Visualization (PCA Space)")
    
    fig = px.scatter(
        df_cluster,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['Model', 'Region', 'Price_USD', 'Sales_Volume'],
        title="Clusters in PCA Space",
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.7
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Profiles
    st.markdown("### Cluster Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster sizes
        cluster_sizes = df_cluster['Cluster'].value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_sizes.index.astype(str),
            y=cluster_sizes.values,
            title="Cluster Sizes",
            labels={'x': 'Cluster', 'y': 'Count'},
            color=cluster_sizes.values,
            color_continuous_scale='Blues',
            text=cluster_sizes.values
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            showlegend=False,
            yaxis=dict(range=[0, cluster_sizes.max() * 1.15]),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by cluster
        cluster_revenue = cluster_profiles['Total_Revenue'].sort_values(ascending=True)
        
        fig = px.bar(
            x=cluster_revenue.values / 1e6,
            y=cluster_revenue.index.astype(str),
            orientation='h',
            title="Total Revenue by Cluster (Millions USD)",
            labels={'x': 'Revenue ($M)', 'y': 'Cluster'},
            color=cluster_revenue.values,
            color_continuous_scale='Greens',
            text=[f'${v/1e6:.0f}M' for v in cluster_revenue.values]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            showlegend=False,
            xaxis=dict(range=[0, (cluster_revenue.max() / 1e6) * 1.2]),
            margin=dict(t=60, b=40, l=60, r=100)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profile table
    st.markdown("### Detailed Cluster Statistics")
    
    display_profiles = cluster_profiles.copy()
    display_profiles['Avg_Price'] = display_profiles['Avg_Price'].apply(lambda x: f'${x:,.0f}')
    display_profiles['Avg_Sales'] = display_profiles['Avg_Sales'].apply(lambda x: f'{x:,.0f}')
    display_profiles['Avg_Mileage'] = display_profiles['Avg_Mileage'].apply(lambda x: f'{x:,.0f} km')
    display_profiles['Total_Revenue'] = display_profiles['Total_Revenue'].apply(lambda x: f'${x/1e6:.1f}M')
    display_profiles['EV_Hybrid_Ratio'] = display_profiles['EV_Hybrid_Ratio'].apply(lambda x: f'{x*100:.1f}%')
    
    st.dataframe(display_profiles, use_container_width=True)
    
    st.markdown("---")
    
    # Distribution across Continents and Models
    st.markdown("### Sales Behavior Patterns: Clusters Across Continents and Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster distribution by continent
        cluster_by_continent = df_cluster.groupby(['Region', 'Cluster']).size().reset_index(name='Count')
        
        fig = px.bar(
            cluster_by_continent,
            x='Region',
            y='Count',
            color='Cluster',
            title="Cluster Distribution Across Continents",
            labels={'Region': 'Continent', 'Count': 'Number of Records'},
            color_discrete_sequence=COLOR_SEQUENCE,
            barmode='group',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='inside', textfont_size=10)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            xaxis_tickangle=-45,
            margin=dict(t=60, b=100, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster distribution by model
        cluster_by_model = df_cluster.groupby(['Model', 'Cluster']).size().reset_index(name='Count')
        
        fig = px.bar(
            cluster_by_model,
            x='Model',
            y='Count',
            color='Cluster',
            title="Cluster Distribution Across Models",
            labels={'Model': 'BMW Model', 'Count': 'Number of Records'},
            color_discrete_sequence=COLOR_SEQUENCE,
            barmode='group',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='inside', textfont_size=10)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            xaxis_tickangle=-45,
            margin=dict(t=60, b=100, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Average sales by continent and cluster
    st.markdown("### Sales Performance by Continent and Cluster")
    
    sales_by_region_cluster = df_cluster.groupby(['Region', 'Cluster'])['Sales_Volume'].mean().reset_index()
    
    fig = px.bar(
        sales_by_region_cluster,
        x='Region',
        y='Sales_Volume',
        color='Cluster',
        title="Average Sales Volume by Continent and Cluster",
        labels={'Region': 'Continent', 'Sales_Volume': 'Avg Sales Volume'},
        color_discrete_sequence=COLOR_SEQUENCE,
        barmode='group',
        text='Sales_Volume'
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside', textfont_size=10)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_tickangle=-45,
        margin=dict(t=60, b=100, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Characteristics
    st.markdown("### Cluster Interpretations")
    
    for cluster_id in range(k):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
        profile = cluster_profiles.loc[cluster_id]
        
        # Determine cluster label based on actual characteristics
        avg_price = profile['Avg_Price'] if isinstance(profile['Avg_Price'], (int, float)) else float(profile['Avg_Price'].replace('$', '').replace(',', ''))
        avg_sales = profile['Avg_Sales'] if isinstance(profile['Avg_Sales'], (int, float)) else float(profile['Avg_Sales'].replace(',', ''))
        avg_year = cluster_data['Year'].mean()
        avg_mileage = cluster_data['Mileage_KM'].mean()
        
        # Create meaningful cluster labels
        if avg_price > 85000 and avg_sales < 5000:
            cluster_label = "Premium Luxury / Low Volume"
        elif avg_price > 85000 and avg_sales >= 5000:
            cluster_label = "Premium Luxury / High Volume"
        elif avg_price > 60000 and avg_sales >= 5000:
            cluster_label = "Mid-Premium / High Volume"
        elif avg_price > 60000 and avg_sales < 5000:
            cluster_label = "Mid-Premium / Low Volume"
        elif avg_sales >= 5000:
            cluster_label = "Entry Level / High Volume"
        else:
            cluster_label = "Entry Level / Low Volume"
        
        with st.expander(f"📊 Cluster {cluster_id}: {cluster_label}", expanded=(cluster_id == 0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Top 5 Models:**")
                top_models = cluster_data['Model'].value_counts().head(5)
                for model, count in top_models.items():
                    pct = (count / len(cluster_data)) * 100
                    st.write(f"- {model}: {count:,} ({pct:.1f}%)")
            
            with col2:
                st.markdown("**Top 5 Continents:**")
                top_regions = cluster_data['Region'].value_counts().head(5)
                for region, count in top_regions.items():
                    pct = (count / len(cluster_data)) * 100
                    st.write(f"- {region}: {count:,} ({pct:.1f}%)")
            
            with col3:
                st.markdown("**Key Characteristics:**")
                st.write(f"- Avg Year: {avg_year:.0f}")
                st.write(f"- Avg Mileage: {avg_mileage:,.0f} km")
                st.write(f"- Avg Price: ${avg_price:,.0f}")
                st.write(f"- Avg Sales: {avg_sales:,.0f}")
                st.write(f"- Size: {len(cluster_data):,} records")
    
    st.markdown("---")
    
    # Answer to Research Question with specific cluster findings
    # Gather specific data about each cluster
    cluster_summaries = []
    for cluster_id in range(k):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
        profile = cluster_profiles.loc[cluster_id]
        
        avg_price = profile['Avg_Price'] if isinstance(profile['Avg_Price'], (int, float)) else float(profile['Avg_Price'].replace('$', '').replace(',', ''))
        avg_sales = profile['Avg_Sales'] if isinstance(profile['Avg_Sales'], (int, float)) else float(profile['Avg_Sales'].replace(',', ''))
        
        top_model = cluster_data['Model'].value_counts().index[0]
        top_continent = cluster_data['Region'].value_counts().index[0]
        size = len(cluster_data)
        
        # Create label
        if avg_price > 85000 and avg_sales < 5000:
            cluster_label = "Premium Luxury / Low Volume"
        elif avg_price > 85000 and avg_sales >= 5000:
            cluster_label = "Premium Luxury / High Volume"
        elif avg_price > 60000 and avg_sales >= 5000:
            cluster_label = "Mid-Premium / High Volume"
        elif avg_price > 60000 and avg_sales < 5000:
            cluster_label = "Mid-Premium / Low Volume"
        elif avg_sales >= 5000:
            cluster_label = "Entry Level / High Volume"
        else:
            cluster_label = "Entry Level / Low Volume"
        
        cluster_summaries.append({
            'id': cluster_id,
            'label': cluster_label,
            'model': top_model,
            'continent': top_continent,
            'price': avg_price,
            'sales': avg_sales,
            'size': size
        })
    
    # Build specific findings text
    specific_findings = []
    for i, summary in enumerate(cluster_summaries):
        specific_findings.append(
            f"**Cluster {summary['id']}: {summary['label']}**\n"
            f"• Dominant Model: {summary['model']}\n"
            f"• Primary Continent: {summary['continent']}\n"
            f"• Avg Price: ${summary['price']:,.0f} | Avg Sales: {summary['sales']:,.0f}\n"
            f"• Size: {summary['size']:,} records ({summary['size']/len(df_cluster)*100:.1f}% of data)"
        )
    
    findings_text = "\n\n".join(specific_findings)
    
    st.success(f"""
**📊 Answer to Research Question:**

**YES, unsupervised learning reveals {k} distinct sales behavior patterns across continents and models:**

The K-Means clustering analysis (optimized via silhouette score = {sil_score:.4f}) identified {k} distinct market segments with clearly different characteristics:

{findings_text}

**Cross-Cluster Insights:**

• Geographic Concentration: Different clusters dominate different continents, showing regional market preferences

• Model Segmentation: Specific BMW models are concentrated in specific clusters, indicating targeted market positioning

• Price-Volume Relationship: Clear inverse relationship between pricing tier and sales volume across clusters

• Market Coverage: The {k} clusters together represent BMW's complete market strategy from luxury exclusivity to volume-driven segments

**Business Implications:**

• Targeted Marketing: Each cluster requires distinct marketing approaches based on their price point and volume characteristics

• Regional Strategy: Continental sales teams should focus on cluster-specific model portfolios that align with local preferences

• Inventory Planning: Production and distribution can be optimized based on cluster-specific demand patterns
    """)


# =============================================================================
# SECTION 7: RESEARCH QUESTION 4
# =============================================================================
@st.cache_data
def forecast_revenue(_df_analysis: pd.DataFrame):
    """Perform revenue forecasting."""
    
    yearly = _df_analysis.groupby('Year').agg({
        'Revenue': 'sum',
        'Sales_Volume': 'sum'
    }).reset_index().sort_values('Year')
    
    yearly['Revenue_Millions'] = yearly['Revenue'] / 1e6
    
    # Simple linear regression for trend
    X_year = yearly['Year'].values
    y_rev = yearly['Revenue'].values
    
    slope, intercept, r_value, p_value, std_err = linregress(X_year, y_rev)
    trend_line = intercept + slope * X_year
    
    # Forecast next year
    next_year = X_year.max() + 1
    forecast = intercept + slope * next_year
    
    # Confidence interval based on residuals
    residuals = y_rev - trend_line
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    lower_ci = forecast - 1.96 * rmse
    upper_ci = forecast + 1.96 * rmse
    
    # Calculate MAPE
    mape = np.mean(np.abs(residuals / y_rev)) * 100
    
    # Year-over-year data
    yearly['YoY_Growth'] = yearly['Revenue'].pct_change() * 100
    
    # Expected growth
    last_year_rev = yearly.iloc[-1]['Revenue']
    expected_growth = ((forecast - last_year_rev) / last_year_rev) * 100
    
    return yearly, trend_line, next_year, forecast, lower_ci, upper_ci, slope, r_value, p_value, mape, expected_growth


def show_rq4(df_analysis):
    """Display Research Question 4 analysis."""
    
    st.markdown("## 🔮 RQ4: Sales Forecasting")
    st.markdown("""
    > **Question:** Can we build a predictive model to forecast BMW sales for the next year with confidence?
    """)
    st.markdown("---")
    
    yearly, trend_line, next_year, forecast, lower_ci, upper_ci, slope, r_value, p_value, mape, expected_growth = forecast_revenue(df_analysis)
    
    # Forecast Summary
    st.markdown("### Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{int(next_year)} Forecast",
            f"${forecast/1e6:.1f}M",
            f"{expected_growth:+.1f}%"
        )
    with col2:
        st.metric("Lower Bound (95% CI)", f"${lower_ci/1e6:.1f}M")
    with col3:
        st.metric("Upper Bound (95% CI)", f"${upper_ci/1e6:.1f}M")
    with col4:
        reliability = "Excellent" if mape < 5 else ("Good" if mape < 10 else ("Fair" if mape < 15 else "Poor"))
        st.metric("Forecast Reliability", reliability)
    
    st.markdown("---")
    
    # Forecast Visualization
    st.markdown("### Revenue Trend & Forecast")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=yearly['Year'],
        y=yearly['Revenue_Millions'],
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color=BMW_COLORS['primary'], width=3),
        marker=dict(size=10)
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=yearly['Year'],
        y=trend_line / 1e6,
        mode='lines',
        name='Trend Line',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Forecast point
    fig.add_trace(go.Scatter(
        x=[next_year],
        y=[forecast / 1e6],
        mode='markers',
        name=f'{int(next_year)} Forecast',
        marker=dict(color='green', size=15, symbol='star')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=[next_year, next_year],
        y=[lower_ci / 1e6, upper_ci / 1e6],
        mode='lines',
        name='95% CI',
        line=dict(color='green', width=4)
    ))
    
    # Annotation
    fig.add_annotation(
        x=next_year,
        y=forecast / 1e6,
        text=f"${forecast/1e6:.1f}M<br>({expected_growth:+.1f}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=50,
        ay=-50,
        font=dict(size=12, color='green'),
        bgcolor='white',
        bordercolor='green',
        borderwidth=1
    )
    
    fig.update_layout(
        title="BMW Revenue Forecast",
        xaxis_title="Year",
        yaxis_title="Revenue (Millions USD)",
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("### Forecast Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("R² Score", f"{r_value**2:.4f}")
            st.metric("MAPE", f"{mape:.2f}%")
        with col1b:
            st.metric("Slope", f"${slope/1e6:.2f}M/year")
            st.metric("Trend p-value", f"{p_value:.4f}")
    
    with col2:
        # Residuals distribution
        residuals = yearly['Revenue'].values - trend_line
        
        fig = px.histogram(
            x=residuals / 1e6,
            nbins=10,
            title="Forecast Residuals Distribution",
            labels={'x': 'Residual ($M)', 'y': 'Frequency'},
            color_discrete_sequence=[BMW_COLORS['secondary']]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Historical Performance Table
    st.markdown("### Historical Revenue & Growth")
    
    display_yearly = yearly.copy()
    display_yearly['Revenue'] = display_yearly['Revenue'].apply(lambda x: f'${x/1e6:.1f}M')
    display_yearly['Revenue_Millions'] = display_yearly['Revenue_Millions'].apply(lambda x: f'{x:.1f}')
    display_yearly['YoY_Growth'] = display_yearly['YoY_Growth'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else '-')
    
    st.dataframe(display_yearly[['Year', 'Revenue', 'YoY_Growth']], use_container_width=True)
    
    st.markdown("---")
    
    # Forecast Reliability Assessment
    st.markdown("### Forecast Reliability Assessment")
    
    st.markdown(f"""
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | R² Score | {r_value**2:.4f} | {'Strong' if r_value**2 > 0.7 else 'Moderate' if r_value**2 > 0.4 else 'Weak'} linear fit |
    | MAPE | {mape:.2f}% | {reliability} accuracy |
    | Trend Significance | p = {p_value:.4f} | {'Statistically significant' if p_value < 0.05 else 'Not significant'} |
    | Expected Growth | {expected_growth:+.1f}% | {'Positive' if expected_growth > 0 else 'Negative'} trend |
    """)


# =============================================================================
# SECTION 8: EXECUTIVE SUMMARY & CONCLUSIONS
# =============================================================================
def show_conclusions(df_analysis):
    """Display executive summary and strategic conclusions with new visualizations."""
    
    st.markdown("## 📋 Executive Summary & Strategic Conclusions")
    st.markdown("---")
    
    # High-level overview
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0066B1 0%, #003d82 100%); padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <h3 style="margin: 0 0 1rem 0; text-align: center;">BMW Global Sales Analysis: Strategic Insights</h3>
        <p style="font-size: 1.05rem; opacity: 0.95; margin: 0; text-align: center; line-height: 1.6;">
            This comprehensive analysis of 50,000 BMW vehicle records (2010-2024) reveals critical insights into pricing strategies, 
            market segmentation, revenue patterns, and growth opportunities across global markets.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Overview with visuals
    st.markdown("### 📊 Analysis Scope & Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df_analysis['Revenue'].sum()
    total_vehicles = len(df_analysis)
    avg_price = df_analysis['Price_USD'].mean()
    num_regions = df_analysis['Region'].nunique()
    
    with col1:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue/1e9:.2f}B",
            delta="15-year period"
        )
    
    with col2:
        st.metric(
            label="Vehicles Analyzed",
            value=f"{total_vehicles:,}",
            delta=f"{df_analysis['Model'].nunique()} models"
        )
    
    with col3:
        st.metric(
            label="Global Avg Price",
            value=f"${avg_price:,.0f}",
            delta="All segments"
        )
    
    with col4:
        st.metric(
            label="Markets Covered",
            value=f"{num_regions} Continents",
            delta="Worldwide"
        )
    
    st.markdown("---")
    
    # Strategic Findings - Comparative Visualizations
    st.markdown("### 🎯 Strategic Findings: Cross-Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by fuel type
        st.markdown("#### Price Positioning by Fuel Type")
        fuel_price = df_analysis.groupby('Fuel_Type')['Price_USD'].agg(['mean', 'median', 'std']).reset_index()
        fuel_price = fuel_price.sort_values('mean', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fuel_price['Fuel_Type'],
            y=fuel_price['mean'],
            name='Average Price',
            marker_color='#0066B1',
            text=fuel_price['mean'],
            texttemplate='$%{text:,.0f}',
            textposition='inside',
            textfont=dict(size=11, color='white')
        ))
        fig.update_layout(
            title="Average Price by Fuel Type",
            xaxis_title="Fuel Type",
            yaxis_title="Average Price (USD)",
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(t=60, b=80, l=60, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales volume trends by region
        st.markdown("#### Regional Sales Distribution")
        region_sales = df_analysis.groupby('Region')['Sales_Volume'].sum().reset_index()
        region_sales = region_sales.sort_values('Sales_Volume', ascending=True)
        
        fig = px.bar(
            region_sales,
            x='Sales_Volume',
            y='Region',
            orientation='h',
            title="Total Sales Volume by Continent",
            labels={'Sales_Volume': 'Total Sales Volume', 'Region': 'Continent'},
            color='Sales_Volume',
            color_continuous_scale='Blues',
            text='Sales_Volume'
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside', textfont_size=11)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            showlegend=False,
            margin=dict(t=60, b=40, l=100, r=60)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue and Price Trends Over Time
    st.markdown("### 📈 Temporal Analysis: Growth Patterns")
    
    yearly_data = df_analysis.groupby('Year').agg({
        'Revenue': 'sum',
        'Price_USD': 'mean',
        'Sales_Volume': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Revenue'] / 1e6,
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#0066B1', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0,102,177,0.2)'
        ))
        fig.update_layout(
            title="Annual Revenue Trend (2010-2024)",
            xaxis_title="Year",
            yaxis_title="Revenue (Millions USD)",
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(t=60, b=40, l=60, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average price trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Price_USD'],
            mode='lines+markers',
            name='Avg Price',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(40,167,69,0.2)'
        ))
        fig.update_layout(
            title="Average Vehicle Price Trend (2010-2024)",
            xaxis_title="Year",
            yaxis_title="Average Price (USD)",
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(t=60, b=40, l=60, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance Comparison
    st.markdown("### 🚗 Model Performance: Top Performers")
    
    model_metrics = df_analysis.groupby('Model').agg({
        'Revenue': 'sum',
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()
    model_metrics['Revenue_M'] = model_metrics['Revenue'] / 1e6
    model_metrics = model_metrics.sort_values('Revenue_M', ascending=False).head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Revenue ($M)',
        x=model_metrics['Model'],
        y=model_metrics['Revenue_M'],
        marker_color='#0066B1',
        yaxis='y',
        offsetgroup=0,
    ))
    
    fig.add_trace(go.Scatter(
        name='Avg Price',
        x=model_metrics['Model'],
        y=model_metrics['Price_USD'],
        marker_color='#ff6b6b',
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Top 10 Models: Revenue vs Average Price',
        xaxis=dict(title='BMW Model', tickangle=-45),
        yaxis=dict(
            title='Revenue (Millions USD)',
            title_font=dict(color='#0066B1'),
            tickfont=dict(color='#0066B1')
        ),
        yaxis2=dict(
            title='Average Price (USD)',
            title_font=dict(color='#ff6b6b'),
            tickfont=dict(color='#ff6b6b'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(t=100, b=120, l=70, r=100),
        legend=dict(
            x=0.5, 
            y=1.15, 
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Strategic Insights Summary
    st.markdown("### 💡 Strategic Insights & Business Implications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(102,126,234,0.4);">
            <div style="display: inline-block; padding: 0.3rem 1rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white; font-weight: 700;">🎯 Pricing Optimization</h4>
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #1a1a2e; font-weight: 700; list-style-position: outside;">
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Year, Mileage, and Engine Size are the key pricing drivers (89.9% R² accuracy)</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Implement ML-based dynamic pricing for real-time optimization</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Regional pricing strategies can capture $7,209 variance across continents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(240,147,251,0.4);">
            <div style="display: inline-block; padding: 0.3rem 1rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white; font-weight: 700;">📊 Market Segmentation</h4>
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #1a1a2e; font-weight: 700; list-style-position: outside;">
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Distinct customer clusters require tailored marketing approaches</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Premium luxury, mid-premium, and entry-level segments clearly defined</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Regional preferences vary significantly across continents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(79,172,254,0.4);">
            <div style="display: inline-block; padding: 0.3rem 1rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white; font-weight: 700;">🌍 Geographic Strategy</h4>
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #1a1a2e; font-weight: 700; list-style-position: outside;">
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">North America and Europe drive total volume</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Middle East commands highest per-vehicle pricing</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Tailor model portfolios to continental cluster preferences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 2rem; border-radius: 15px; box-shadow: 0 8px 24px rgba(67,233,123,0.4);">
            <div style="display: inline-block; padding: 0.3rem 1rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white; font-weight: 700;">📈 Revenue Growth</h4>
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #1a1a2e; font-weight: 700; list-style-position: outside;">
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Forecast models enable reliable budget planning with 95% confidence intervals</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">Top 10 models generate majority of revenue - focus on winners</li>
                <li style="text-shadow: 1px 1px 2px rgba(255,255,255,0.2);">SUV and Performance categories show strongest revenue potential</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology Summary
    st.markdown("### 📚 Analytical Approach & Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Methods Applied:**
        - 🔬 **Machine Learning:** Random Forest (R² = 89.9%), Gradient Boosting, Ridge Regression
        - 📊 **Clustering:** K-Means optimized via silhouette score
        - 📈 **Statistical Testing:** ANOVA (F=57.3, p<0.001), Shapiro-Wilk normality
        - 🔮 **Forecasting:** Linear regression with 95% confidence intervals
        - ✅ **Validation:** Cross-validation, train-test splits, multiple metrics
        
        **Data Quality:**
        - ✓ 50,000 records across 15 years (2010-2024)
        - ✓ Multiple continents and BMW model types
        - ✓ Comprehensive feature set (price, sales, mileage, year, engine, fuel)
        """)
    
    with col2:
        st.markdown("""
        **Key Strengths:**
        - ✅ Large dataset ensures statistical reliability
        - ✅ Multiple analytical methods provide cross-validation
        - ✅ High model accuracy (R² = 89.9%) confirms robust findings
        - ✅ Clear actionable insights for business strategy
        
        **Assumptions & Limitations:**
        - ⚠️ Historical data - assumes similar market conditions continue
        - ⚠️ External factors (economic shocks, competition) not modeled
        - ⚠️ Limited to available features in dataset
        - ⚠️ Forecasts require ongoing monitoring and adjustment
        """)
    
    st.markdown("---")
    
    # Final Recommendations
    st.success("""
**🚀 Immediate Action Items:**

1. **Deploy ML Pricing Model:** Implement Year/Mileage/Engine-based dynamic pricing system

2. **Regional Portfolio Optimization:** Adjust model mix based on continental cluster analysis findings

3. **Segment-Specific Marketing:** Launch tailored campaigns for premium luxury, mid-premium, and entry-level clusters

4. **Revenue Monitoring:** Integrate forecast model into quarterly planning and variance tracking

**Long-Term Strategic Initiatives:**

• Expand data collection to include competitive intelligence and economic indicators

• Develop real-time dashboard for ongoing performance monitoring against forecasts

• Invest in high-revenue categories (SUV, Performance) identified in top 10 models analysis

• Refine cluster-based inventory and production planning for regional markets
    """)
    
    st.markdown("---")
    
    # Thank You
    st.markdown("""
    <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; color: white; margin-top: 2rem;">
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">Thank You!</h2>
        <p style="font-size: 1.2rem; opacity: 0.95; margin: 0 0 1.5rem 0;">
            BMW Global Sales Analysis - Group 4
        </p>
        <p style="font-size: 1.1rem; opacity: 0.85; margin: 0 0 0.5rem 0;">
            Questions? Comments? Let's discuss!
        </p>
        <p style="font-size: 0.95rem; opacity: 0.7; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
            Research Team: Marcel Klibansky • Kevin Torano • Zachary Bramwell • Bernardo Sastre
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Sidebar
    st.sidebar.markdown("## 🚗 BMW Sales Analysis")
    st.sidebar.markdown("---")
    
    # Load data (removed file uploader)
    try:
        df = load_data()
        df_analysis = engineer_features(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure 'BMW_Final_Data.csv' is in the working directory.")
        return
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📑 Navigation")
    
    sections = [
        "🏠 Introduction",
        "📁 Data Overview",
        "🔍 Exploratory Data Analysis",
        "🔬 RQ1: Key Factors",
        "💰 RQ2: Revenue Analysis",
        "🎯 RQ3: Clustering",
        "🔮 RQ4: Forecasting",
        "📋 Executive Summary & Conclusions"
    ]
    
    selected_section = st.sidebar.radio("Go to:", sections, index=0)
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown(f"""
    - **Records:** {len(df):,}
    - **Years:** {df['Year'].min()} - {df['Year'].max()}
    - **Regions:** {df['Region'].nunique()}
    - **Models:** {df['Model'].nunique()}
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 Analysis run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Render selected section
    if selected_section == "🏠 Introduction":
        show_introduction(df, df_analysis)
    elif selected_section == "📁 Data Overview":
        show_data_overview(df, df_analysis)
    elif selected_section == "🔍 Exploratory Data Analysis":
        show_eda(df, df_analysis)
    elif selected_section == "🔬 RQ1: Key Factors":
        show_rq1(df_analysis)
    elif selected_section == "💰 RQ2: Revenue Analysis":
        show_rq2(df_analysis)
    elif selected_section == "🎯 RQ3: Clustering":
        show_rq3(df_analysis)
    elif selected_section == "🔮 RQ4: Forecasting":
        show_rq4(df_analysis)
    elif selected_section == "📋 Executive Summary & Conclusions":
        show_conclusions(df_analysis)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import matplotlib.colors

# Set page config
st.set_page_config(page_title="Netflix Analysis App", layout="wide")

# Convert Seaborn palette to hex colors for Plotly compatibility
COLOR_PALETTE = [matplotlib.colors.to_hex(color) for color in sns.color_palette("Set2")]

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; padding: 20px; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-heading { font-family: 'Arial', sans-serif; font-size: 2.5rem; color: #1f2a44; overflow: hidden; white-space: nowrap; }
    .marquee { display: inline-block; animation: marquee 10s linear infinite; }
    @keyframes marquee { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    h2 { color: #1f2a44; font-family: 'Arial', sans-serif; transition: all 0.3s ease; }
    h2:hover { color: #e50914; transform: scale(1.05); }
    h3 { color: #1f2a44; font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background-color: #1f2a44; color: white; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("""
    <div class="main-heading">
        <span class="marquee">Netflix Data Analysis Dashboard</span>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
section = st.sidebar.selectbox("Choose a Section", [
    "Home",
    "TV Shows vs Movies Comparison",
    "Dataset Overview",
    "Content Added per Year",
    "Rating Distribution",
    "Top 10 Countries",
    "Most Common Genres"
], index=0)

# Load and Clean Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("netflix_titles.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def clean_data(df):
    try:
        df = df.drop_duplicates()
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
        df['director'] = df['director'].fillna('Unknown')
        df['cast'] = df['cast'].fillna('Unknown')
        df['country'] = df['country'].fillna('Unknown').str.split(', ')
        df['rating'] = df['rating'].fillna('Not Rated')
        df['duration'] = df['duration'].fillna('Unknown')
        df['listed_in'] = df['listed_in'].fillna('Unknown').str.split(', ')
        # Enhanced duration extraction
        df['duration_num'] = df['duration'].str.extract(r'(\d+)\s*(min|Season|Seasons)?', expand=True)[0].astype(float)
        df['duration_unit'] = df['duration'].str.extract(r'(\d+)\s*(min|Season|Seasons)?', expand=True)[1]
        # Impute missing duration_num
        movie_median = df[df['type'] == 'Movie']['duration_num'].median()
        tv_median = df[df['type'] == 'TV Show']['duration_num'].median()
        df.loc[(df['type'] == 'Movie') & (df['duration_num'].isna()), 'duration_num'] = movie_median
        df.loc[(df['type'] == 'TV Show') & (df['duration_num'].isna()), 'duration_num'] = tv_median
        # Log invalid durations
        invalid_duration = df[df['duration_num'].isna() & (df['duration'] != 'Unknown')]
        if not invalid_duration.empty:
            st.warning(f"Found {len(invalid_duration)} rows with invalid duration formats. Examples: {invalid_duration['duration'].head().tolist()}")
        # Create rating_num column
        rating_map = {
            'TV-Y': 1, 'TV-Y7': 2, 'TV-G': 3, 'TV-PG': 4, 'TV-14': 7,
            'TV-MA': 9, 'R': 9, 'PG-13': 7, 'PG': 5, 'G': 3,
            'NC-17': 9, 'Not Rated': 0
        }
        df['rating_num'] = df['rating'].map(rating_map)
        rating_median = df['rating_num'].median()
        df['rating_num'] = df['rating_num'].fillna(rating_median)
        return df
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return df

raw_df = load_data()
df = clean_data(raw_df)

# Helper Functions
def descriptive_stats_and_ci(data, label, unit="", bootstrap=False):
    try:
        mean = data.mean()
        std = data.std()
        n = len(data)
        desc_stats = pd.DataFrame(data.describe())
        if bootstrap:
            boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(1000)]
            ci = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
        stat, p = stats.shapiro(data) if n <= 5000 else (0, 1)
        normality_text = f"Shapiro-Wilk p-value: {p:.4f} ({'non-normal' if p < 0.05 else 'normal'})"
        return desc_stats, mean, ci, f"{label}: ({ci[0]:.2f}, {ci[1]:.2f}) {unit}", normality_text
    except Exception as e:
        st.error(f"Error in descriptive stats for {label}: {str(e)}")
        return pd.DataFrame(), 0, (0, 0), "", ""

def plot_histogram(data, label, unit="", col=None):
    try:
        if col:
            with col:
                fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                sns.histplot(data, bins=20, kde=True, ax=ax_hist, color=COLOR_PALETTE[0])
                ax_hist.set_xlabel(f"{label} ({unit})")
                ax_hist.set_ylabel("Count")
                ax_hist.set_title(f"Distribution of {label}")
                peak_value = data.mode()[0] if not data.mode().empty else data.mean()
                peak_count = len(data[data == peak_value])
                total_count = len(data)
                proportion_peak = peak_count / total_count * 100
                plt.tight_layout()
                st.pyplot(fig_hist)
                st.markdown(f"**Insight**: The distribution peaks at **{peak_value}** with **{proportion_peak:.2f}%** of the data.")
                return peak_value, proportion_peak
        else:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
            sns.histplot(data, bins=20, kde=True, ax=ax_hist, color=COLOR_PALETTE[0])
            ax_hist.set_xlabel(f"{label} ({unit})")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(f"Distribution of {label}")
            peak_value = data.mode()[0] if not data.mode().empty else data.mean()
            peak_count = len(data[data == peak_value])
            total_count = len(data)
            proportion_peak = peak_count / total_count * 100
            plt.tight_layout()
            st.pyplot(fig_hist)
            st.markdown(f"**Insight**: The distribution peaks at **{peak_value}** with **{proportion_peak:.2f}%** of the data.")
            return peak_value, proportion_peak
    except Exception as e:
        st.error(f"Error plotting histogram for {label}: {str(e)}")
        return None, None

def plot_ci_bar(mean, ci, label, unit="", col=None):
    try:
        if col:
            with col:
                fig_ci, ax_ci = plt.subplots(figsize=(8, 6))
                ax_ci.bar([label], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=10, color=COLOR_PALETTE[1])
                ax_ci.set_ylabel(f"{label} ({unit})")
                ax_ci.set_title(f"95% Confidence Interval for {label}")
                plt.tight_layout()
                st.pyplot(fig_ci)
                st.markdown(f"**Insight**: The 95% confidence interval for **{label}** is **({ci[0]:.2f}, {ci[1]:.2f})** {unit}.")
                return mean, ci
        else:
            fig_ci, ax_ci = plt.subplots(figsize=(8, 6))
            ax_ci.bar([label], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=10, color=COLOR_PALETTE[1])
            ax_ci.set_ylabel(f"{label} ({unit})")
            ax_ci.set_title(f"95% Confidence Interval for {label}")
            plt.tight_layout()
            st.pyplot(fig_ci)
            st.markdown(f"**Insight**: The 95% confidence interval for **{label}** is **({ci[0]:.2f}, {ci[1]:.2f})** {unit}.")
            return mean, ci
    except Exception as e:
        st.error(f"Error plotting CI for {label}: {str(e)}")
        return None, None

def plot_correlation_matrix(df, numerical_cols, section_name):
    try:
        st.subheader("🔗 Correlation Matrix")
        corr_df = df[numerical_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax_corr, vmin=-1, vmax=1)
        ax_corr.set_title(f"Correlation Matrix ({section_name})")
        max_corr = corr_df.abs().max().max()
        corr_pair = corr_df.abs().stack().idxmax()
        plt.tight_layout()
        st.pyplot(fig_corr)

        # Provide unique insights for each section
        if section_name == "TV Shows vs Movies Comparison":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This indicates a strong relationship between these variables in the context of TV shows vs movies comparison.")
        elif section_name == "Dataset Overview":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This suggests a significant relationship between these variables in the overall dataset.")
        elif section_name == "Content Added per Year":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This highlights a notable relationship between these variables in the context of content added per year.")
        elif section_name == "Rating Distribution":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This indicates a strong relationship between these variables in the context of rating distribution.")
        elif section_name == "Top 10 Countries":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This suggests a significant relationship between these variables in the context of the top 10 countries with the most content.")
        elif section_name == "Most Common Genres":
            st.markdown(f"**Insight**: The strongest correlation is between **{corr_pair[0]}** and **{corr_pair[1]}** with a value of **{max_corr:.2f}**. This indicates a strong relationship between these variables in the context of the most common genres.")
        return max_corr, corr_pair
    except Exception as e:
        st.error(f"Error plotting correlation matrix: {str(e)}")
        return None, None

def regression_analysis(df, section_name, is_movie=True):
    try:
        # Drop rows with NaN in the required columns
        df_clean = df.dropna(subset=['duration_num', 'release_year', 'rating_num'])
        if len(df_clean) < 5:  # Need at least 5 rows to split
            st.warning(f"Insufficient data for regression in {section_name}: only {len(df_clean)} rows available after cleaning.")
            return None, None, None, None
        
        if is_movie:
            X = df_clean[['release_year', 'rating_num']]
            y = df_clean['duration_num']
            target = "Duration (minutes)"
        else:
            X = df_clean[['release_year', 'duration_num']]
            y = df_clean['rating_num']
            target = "Rating Score"
        
        # Check for consistent sample sizes
        if len(X) != len(y):
            st.error(f"Data mismatch in {section_name}: X has {len(X)} samples, y has {len(y)} samples.")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"*Model*: Predicting {target} using release year and {'rating' if is_movie else 'duration'}")
        st.write(f"*R²*: {r2:.4f} (proportion of variance explained)")
        st.write(f"*RMSE*: {rmse:.2f} {'minutes' if is_movie else 'rating units'} (prediction error)")

        # Scatterplot
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
            ax_scatter.scatter(X_test['release_year'], y_test, color=COLOR_PALETTE[2], label='Actual')
            ax_scatter.scatter(X_test['release_year'], y_pred, color=COLOR_PALETTE[3], label='Predicted', alpha=0.5)
            ax_scatter.set_xlabel("Release Year")
            ax_scatter.set_ylabel(target)
            ax_scatter.set_title(f"{target} vs Release Year")
            ax_scatter.legend()
            plt.tight_layout()
            st.pyplot(fig_scatter)
            trend_direction = "increasing" if np.corrcoef(X_test['release_year'], y_test)[0, 1] > 0 else "decreasing"

        # Residual Plot
        with col2:
            fig_resid, ax_resid = plt.subplots(figsize=(8, 6))
            residuals = y_test - y_pred
            ax_resid.scatter(X_test['release_year'], residuals, color=COLOR_PALETTE[4])
            ax_resid.axhline(0, color='red', linestyle='--')
            ax_resid.set_xlabel("Release Year")
            ax_resid.set_ylabel("Residuals")
            ax_resid.set_title("Residual Plot")
            plt.tight_layout()
            st.pyplot(fig_resid)
            residual_mean = np.mean(residuals)

        return trend_direction, r2, rmse, residual_mean
    except Exception as e:
        st.error(f"Error in regression analysis for {section_name}: {str(e)}")
        return None, None, None, None

# ---------- SECTION 1: Home ----------
if section == "Home":
    st.header("🏠 Welcome to the Netflix Data Analysis Dashboard")
    
    st.markdown("""
    This interactive dashboard allows you to explore and analyze Netflix's vast collection of TV shows and movies up to the year 2021.
    It leverages real-world data to uncover meaningful insights and trends, helping users understand Netflix's content landscape.
    
    ### 📌 Objective
    The primary objective of this project is to:
    - Analyze the distribution and evolution of Netflix content over the years.
    - Visualize key statistics like content types, genres, release years, ratings, and countries of origin.
    - Support data-driven decisions or exploratory learning for researchers, data enthusiasts, and casual viewers.

    Use the sidebar to navigate through various analytical sections containing charts, filters, and summaries.
    """)

    st.markdown("[📂 View Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)", unsafe_allow_html=True)

    st.subheader("🔍 View Dataset")

    view_data_option = st.radio("Choose which data to view:", ("Raw Data", "Cleaned Data"))

    if view_data_option == "Raw Data":
        st.markdown("### 📄 Raw Dataset Preview")
        st.dataframe(raw_df)
    else:
        st.markdown("### 📊 Cleaned Dataset Preview")
        st.dataframe(df)

# ---------- SECTION 2: TV Shows vs Movies Comparison ----------
elif section == "TV Shows vs Movies Comparison":
    st.header("📺 TV Shows vs Movies Comparison")

    # Pie Chart: Content Type Distribution
    st.subheader("📊 Content Type Distribution")
    type_counts = df['type'].value_counts()
    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Content Type Distribution",
                 color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)
    movie_prop = type_counts['Movie'] / df.shape[0] * 100
    st.write("*Table: Content Type Counts*")
    st.dataframe(type_counts.rename("Count").to_frame())
    st.markdown(f"**Insight**: Movies constitute **{movie_prop:.1f}%** of the total content, indicating a higher focus on movies compared to TV shows.")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(df, numerical_cols, "TV Shows vs Movies Comparison")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Duration)")
    movie_df = df[df['type'] == 'Movie']['duration_num'].dropna()
    tv_df = df[df['type'] == 'TV Show']['duration_num'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_df.describe(),
        'TV Shows': tv_df.describe()
    })
    st.write("*Summary Statistics for Duration*")
    st.dataframe(desc_stats)
    movie_mean = movie_df.mean()
    movie_range = (movie_df.min(), movie_df.max())
    tv_mean = tv_df.mean()
    st.markdown(f"**Insight**: Movies have an average duration of **{movie_mean:.0f} minutes** (range: **{movie_range[0]:.0f}** - **{movie_range[1]:.0f}** minutes), while TV shows have an average duration of **{tv_mean:.0f} seasons**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Duration)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_df, "Movies", col1, "minutes"), (tv_df, "TV Shows", col2, "seasons")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_df, "Movies", col1, "minutes", 120), (tv_df, "TV Shows", col2, "seasons", 2)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            lambda_param = data.mean()
            prob = stats.poisson.pmf(threshold, lambda_param)
            tv_prob = prob
  
    st.markdown(f"""
    **Insight**: The distribution of movie durations shows a peak at approximately **{movie_hist[0]:.0f} minutes**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that duration. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies exceed 120 minutes, indicating a substantial presence of long-form content. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} seasons**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show having exactly **2 seasons** is **{tv_prob*100:.2f}%**, suggesting most shows either stay short or expand significantly.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    reg_results = regression_analysis(df.dropna(subset=['duration_num', 'release_year', 'rating_num']), "TV Shows vs Movies", is_movie=True)
    if reg_results:
        trend_dir, r2, rmse, residual_mean = reg_results
        # Ensure rmse and r2 are floats
        if isinstance(r2, tuple):
            r2 = r2[0]
        if isinstance(rmse, tuple):
            rmse = rmse[0]
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)

    # Future Trend
    st.subheader("📅 Future Trend")
    yearly_counts = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    years = yearly_counts.index.astype(int)
    movie_counts = yearly_counts.get('Movie', pd.Series(0, index=years))
    tv_counts = yearly_counts.get('TV Show', pd.Series(0, index=years))
    X = years.values.reshape(-1, 1)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    movie_model = LinearRegression().fit(X, movie_counts)
    tv_model = LinearRegression().fit(X, tv_counts)
    movie_future = movie_model.predict(future_years)
    tv_future = tv_model.predict(future_years)
    movie_future = np.maximum(movie_future, 0)
    tv_future = np.maximum(tv_future, 0)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    ax_trend.plot(years, movie_counts, label='Movies', marker='o', color=COLOR_PALETTE[2])
    ax_trend.plot(years, tv_counts, label='TV Shows', marker='o', color=COLOR_PALETTE[3])
    ax_trend.plot(future_years.flatten(), movie_future, linestyle='--', color=COLOR_PALETTE[2])
    ax_trend.plot(future_years.flatten(), tv_future, linestyle='--', color=COLOR_PALETTE[3])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Title Counts")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)
    peak_year_movie = int(movie_counts.idxmax())
    peak_count_movie = int(movie_counts.max())
    movie_future_2030 = int(movie_future[-1])
    tv_future_2030 = int(tv_future[-1])
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    year_idx = np.where(future_years.flatten() == pred_year)[0][0]
    pred_movie = int(movie_future[year_idx])
    pred_tv = int(tv_future[year_idx])
    st.markdown(f"**Insight**: The model projects **{pred_movie:,}** movies and **{pred_tv:,}** TV shows by **{pred_year}**, with a peak of **{peak_count_movie}** movies in **{peak_year_movie}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"Movies constitute {movie_prop:.1f}% of titles, with a peak correlation of {corr_max:.2f} between {corr_pair[0]} and {corr_pair[1]}. Movie durations average {movie_mean:.0f} minutes (range: {movie_range[0]:.0f}-{movie_range[1]:.0f}), while TV shows average {tv_mean:.0f} seasons. Confidence intervals show movie durations at {movie_stats[0]:.0f} minutes (CI: {movie_stats[1][0]:.0f}-{movie_stats[1][1]:.0f}) and TV shows at {tv_stats[0]:.0f} seasons (CI: {tv_stats[1][0]:.0f}-{tv_stats[1][1]:.0f}). Histograms peak at {movie_hist[0]:.0f} minutes ({movie_hist[1]:.1f}% movies) and {tv_hist[0]:.0f} seasons ({tv_hist[1]:.1f}% TV shows), with {movie_prob*100:.1f}% movies exceeding 120 minutes. Regression shows durations from {actual_range[0]:.0f} to {actual_range[1]:.0f} minutes with a {trend_dir} trend, and future projections peak at {peak_count_movie} movies in {peak_year_movie}, reaching {pred_movie:,} movies and {pred_tv:,} TV shows by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"Movies dominate at {movie_prop:.1f}%, with a {corr_max:.2f} correlation between {corr_pair[0]} and {corr_pair[1]}. Durations average {movie_stats[0]:.0f} minutes for movies and {tv_stats[0]:.0f} seasons for TV shows, with {movie_prob*100:.1f}% movies over 120 minutes. Regression confirms a {trend_dir} trend (average {mean_actual:.0f} minutes). Projections suggest {pred_movie:,} movies and {pred_tv:,} TV shows by {pred_year}, peaking at {peak_count_movie} in {peak_year_movie}.")

elif section == "Dataset Overview":
    st.header("📋 Dataset Overview")

    # Bar Chart: Titles by Type
    st.subheader("📊 Titles by Type")
    type_counts = df['type'].value_counts()
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette=COLOR_PALETTE, ax=ax_bar)
    ax_bar.set_xlabel("Content Type")
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Titles by Type")
    plt.tight_layout()
    st.pyplot(fig_bar)
    movie_count = type_counts['Movie']
    total_titles = df.shape[0]
    movie_prop = movie_count / total_titles * 100
    st.write("*Table: Dataset Summary*")
    st.dataframe(type_counts.rename("Count").to_frame())
    st.markdown(f"**Insight**: Movies account for **{movie_count}** of **{total_titles}** titles (**{movie_prop:.1f}%**).")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(df, numerical_cols, "Dataset Overview")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Release Year)")
    movie_year_data = df[df['type'] == 'Movie']['release_year'].dropna()
    tv_year_data = df[df['type'] == 'TV Show']['release_year'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_year_data.describe(),
        'TV Shows': tv_year_data.describe()
    })
    st.write("*Summary Statistics for Release Year*")
    st.dataframe(desc_stats)
    movie_mean = movie_year_data.mean()
    movie_range = (movie_year_data.min(), movie_year_data.max())
    tv_mean = tv_year_data.mean()
    st.markdown(f"**Insight**: Movie release years average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), while TV show release years average **{tv_mean:.0f}**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Release Year)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_year_data, "Movies", col1, "years"), (tv_year_data, "TV Shows", col2, "years")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_year_data, "Movies", col1, "years", 2015), (tv_year_data, "TV Shows", col2, "years", 2015)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            tv_prob = prob
    st.markdown(f"""
    **Insight**: The distribution of movie release years shows a peak at approximately **{movie_hist[0]:.0f} years**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that year. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies were released after 2015, indicating a focus on recent content. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} years**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show being released after 2015 is **{tv_prob*100:.2f}%**, reflecting a similar trend towards newer releases.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    reg_results = regression_analysis(df.dropna(subset=['duration_num', 'release_year', 'rating_num']), "Dataset Overview", is_movie=True)
    if reg_results:
        trend_dir, r2, rmse, residual_mean = reg_results
        if isinstance(r2, tuple):
            r2 = r2[0]
        if isinstance(rmse, tuple):
            rmse = rmse[0]
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)

    # Future Trend
    st.subheader("📅 Future Trend")
    yearly_counts = df['year_added'].value_counts().sort_index()
    years = yearly_counts.index.astype(int)
    counts = yearly_counts.values
    X = years.values.reshape(-1, 1)
    model = LinearRegression().fit(X, counts)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    future_counts = model.predict(future_years)
    future_counts = np.maximum(future_counts, 0)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    ax_trend.plot(years, counts, label='Historical', marker='o', color=COLOR_PALETTE[4])
    ax_trend.plot(future_years.flatten(), future_counts, label='Projected', linestyle='--', color=COLOR_PALETTE[4])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Title Counts")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)
    peak_year = int(years[counts.argmax()])
    peak_count = int(counts.max())
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    year_idx = np.where(future_years.flatten() == pred_year)[0][0]
    pred_titles = int(future_counts[year_idx])
    st.markdown(f"**Insight**: The model projects **{pred_titles:,}** titles by **{pred_year}**, with a peak of **{peak_count}** titles in **{peak_year}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"Movies account for **{movie_count}** of **{total_titles}** titles (**{movie_prop:.1f}%**), with a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Release years average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), with a CI of **{movie_stats[0]:.0f}** (CI: **{movie_stats[1][0]:.0f}**-**{movie_stats[1][1]:.0f}**). The histogram peaks at **{movie_hist[0]:.0f}** (**{movie_hist[1]:.1f}%** of titles), and **{movie_prob*100:.1f}%** are post-2015. Regression shows durations from **{actual_range[0]:.0f}** to **{actual_range[1]:.0f}** minutes with a **{trend_dir}** trend, peaking at **{peak_count}** titles in **{peak_year}** and projecting **{pred_titles:,}** by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"Movies dominate with **{movie_prop:.1f}%**, showing a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Release years average **{movie_stats[0]:.0f}**, with **{movie_prob*100:.1f}%** post-2015. Regression confirms a **{trend_dir}** trend (average **{mean_actual:.0f}** minutes), projecting **{pred_titles:,}** titles by {pred_year} after a **{peak_count}** peak in **{peak_year}**.")

# ---------- SECTION 4: Content Added per Year ----------
elif section == "Content Added per Year":
    st.header("📅 Content Added per Year")

    # Line Chart: Content Added per Year
    st.subheader("📊 Content Added per Year")
    yearly_counts = df['year_added'].value_counts().sort_index()
    years = yearly_counts.index.astype(int)
    fig_line, ax_line = plt.subplots(figsize=(8, 6))
    ax_line.plot(years, yearly_counts.values, marker='o', color=COLOR_PALETTE[1])
    ax_line.set_xlabel("Year")
    ax_line.set_ylabel("Number of Titles")
    ax_line.set_title("Content Added per Year")
    plt.tight_layout()
    st.pyplot(fig_line)
    peak_year = int(yearly_counts.idxmax())
    peak_count = int(yearly_counts.max())
    last_year_count = int(yearly_counts.loc[years[-1]])
    st.write("*Table: Content Added per Year*")
    st.dataframe(yearly_counts.rename("Count").to_frame())
    st.markdown(f"**Insight**: Content additions peaked at **{peak_count}** in **{peak_year}**, dropping to **{last_year_count}** in **{years[-1]}**.")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(df, numerical_cols, "Content Added per Year")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Year Added)")
    movie_year_data = df[df['type'] == 'Movie']['year_added'].dropna()
    tv_year_data = df[df['type'] == 'TV Show']['year_added'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_year_data.describe(),
        'TV Shows': tv_year_data.describe()
    })
    st.write("*Summary Statistics for Year Added*")
    st.dataframe(desc_stats)
    movie_mean = movie_year_data.mean()
    movie_range = (movie_year_data.min(), movie_year_data.max())
    tv_mean = tv_year_data.mean()
    st.markdown(f"**Insight**: Movie years added average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), while TV show years added average **{tv_mean:.0f}**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Year Added)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_year_data, "Movies", col1, "years"), (tv_year_data, "TV Shows", col2, "years")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_year_data, "Movies", col1, "years", 2018), (tv_year_data, "TV Shows", col2, "years", 2018)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            tv_prob = prob
    st.markdown(f"""
    **Insight**: The distribution of movie years added shows a peak at approximately **{movie_hist[0]:.0f} years**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that year. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies were added after 2018, indicating a focus on recent additions. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} years**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show being added after 2018 is **{tv_prob*100:.2f}%**, reflecting a similar trend towards recent additions.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    reg_results = regression_analysis(df.dropna(subset=['duration_num', 'release_year', 'rating_num']), "Content Added per Year", is_movie=True)
    if reg_results:
        trend_dir, r2, rmse, residual_mean = reg_results
        if isinstance(r2, tuple):
            r2 = r2[0]
        if isinstance(rmse, tuple):
            rmse = rmse[0]
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)

    # Future Trend
    st.subheader("📅 Future Trend")
    counts = yearly_counts.values
    X = years.values.reshape(-1, 1)
    model = LinearRegression().fit(X, counts)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    future_counts = model.predict(future_years)
    future_counts = np.maximum(future_counts, 0)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    ax_trend.plot(years, counts, label='Historical', marker='o', color=COLOR_PALETTE[1])
    ax_trend.plot(future_years.flatten(), future_counts, label='Projected', linestyle='--', color=COLOR_PALETTE[1])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Title Additions")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)
    future_2030 = int(future_counts[-1])
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    year_idx = np.where(future_years.flatten() == pred_year)[0][0]
    pred_titles = int(future_counts[year_idx])
    st.markdown(f"**Insight**: The model projects **{pred_titles:,}** titles by **{pred_year}**, with a peak of **{peak_count}** titles in **{peak_year}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"Title additions peak at **{peak_count}** in **{peak_year}**, dropping to **{last_year_count}** in **{years[-1]}**, with a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Years added average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), with a CI of **{movie_stats[0]:.0f}** (CI: **{movie_stats[1][0]:.0f}**-**{movie_stats[1][1]:.0f}**). The histogram peaks at **{movie_hist[0]:.0f}** (**{movie_hist[1]:.1f}%** of titles), and **{movie_prob*100:.1f}%** are post-2018. Regression shows durations from **{actual_range[0]:.0f}** to **{actual_range[1]:.0f}** minutes with a **{trend_dir}** trend, projecting **{pred_titles:,}** titles by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"Additions peaked at **{peak_count}** in **{peak_year}**, declining to **{last_year_count}** by **{years[-1]}**, with years averaging **{movie_stats[0]:.0f}** and **{movie_prob*100:.1f}%** post-2018. Regression confirms a **{trend_dir}** trend (average **{mean_actual:.0f}** minutes), projecting **{pred_titles:,}** titles by {pred_year}.")

# ---------- SECTION 5: Rating Distribution ----------
elif section == "Rating Distribution":
    st.header("🔠 Rating Distribution")

    # Pie Chart: Top Ratings
    st.subheader("📊 Top Ratings")
    rating_counts = df['rating'].value_counts().head(5)
    fig = px.pie(values=rating_counts.values, names=rating_counts.index, title="Top 5 Ratings",
                 color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)
    top_rating = rating_counts.idxmax()
    top_prop = rating_counts.max() / df.shape[0] * 100
    total_titles = df.shape[0]
    st.write("*Table: Top 5 Ratings*")
    st.dataframe(rating_counts.rename("Count").to_frame())
    st.markdown(f"**Insight**: **{top_rating}** leads with **{top_prop:.1f}%** of **{total_titles}** titles.")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(df, numerical_cols, "Rating Distribution")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Rating Score)")
    rating_data = df['rating_num'].dropna()
    rating_data = rating_data[rating_data != 0]
    movie_rating_data = df[(df['type'] == 'Movie') & (df['rating_num'] != 0)]['rating_num'].dropna()
    tv_rating_data = df[(df['type'] == 'TV Show') & (df['rating_num'] != 0)]['rating_num'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_rating_data.describe(),
        'TV Shows': tv_rating_data.describe()
    })
    st.write("*Summary Statistics for Rating Score*")
    st.dataframe(desc_stats)
    movie_mean = movie_rating_data.mean()
    movie_range = (movie_rating_data.min(), movie_rating_data.max())
    tv_mean = tv_rating_data.mean()
    st.markdown(f"**Insight**: Movie ratings average **{movie_mean:.1f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), while TV show ratings average **{tv_mean:.1f}**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Rating Score)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_rating_data, "Movies", col1, "score"), (tv_rating_data, "TV Shows", col2, "score")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_rating_data, "Movies", col1, "score", 7), (tv_rating_data, "TV Shows", col2, "score", 7)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            prob = stats.norm.cdf(threshold, loc=mu, scale=sigma)
            tv_prob = prob
    st.markdown(f"""
    **Insight**: The distribution of movie ratings shows a peak at approximately **{movie_hist[0]:.0f} score**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that rating. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies have a rating of TV-14 or lower. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} score**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show having a rating of TV-14 or lower is **{tv_prob*100:.2f}%**.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    reg_results = regression_analysis(df.dropna(subset=['duration_num', 'release_year', 'rating_num']), "Rating Distribution", is_movie=True)
    if reg_results:
        trend_dir, r2, rmse, residual_mean = reg_results
        if isinstance(r2, tuple):
            r2 = r2[0]
        if isinstance(rmse, tuple):
            rmse = rmse[0]
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)

    # Future Trend
    st.subheader("📅 Future Trend")
    top_ratings = rating_counts.index[:3]
    yearly_rating_counts = df[df['rating'].isin(top_ratings)].groupby(['year_added', 'rating']).size().unstack(fill_value=0)
    years = yearly_rating_counts.index.astype(int)
    X = years.values.reshape(-1, 1)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    for rating in top_ratings:
        counts = yearly_rating_counts.get(rating, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_counts = model.predict(future_years)
        future_counts = np.maximum(future_counts, 0)
        ax_trend.plot(years, counts, label=rating, marker='o', color=COLOR_PALETTE[list(top_ratings).index(rating) % len(COLOR_PALETTE)])
        ax_trend.plot(future_years.flatten(), future_counts, linestyle='--', color=COLOR_PALETTE[list(top_ratings).index(rating) % len(COLOR_PALETTE)])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Titles by Rating")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)

    # Sum ratings per year and find peak year
    yearly_totals = yearly_rating_counts[top_ratings].sum(axis=1)
    peak_year_rating = int(yearly_totals.idxmax())
    peak_count_rating = int(yearly_totals.max())
    dominant_rating = yearly_rating_counts[top_ratings].loc[peak_year_rating].idxmax()
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    pred_ratings = {}
    for rating in top_ratings:
        counts = yearly_rating_counts.get(rating, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_count = model.predict([[pred_year]])[0]
        future_count = max(future_count, 0)
        pred_ratings[rating] = int(future_count)
    st.markdown(f"**Insight**: The model projects **{pred_ratings[dominant_rating]:,}** **{dominant_rating}** titles by **{pred_year}**, with a peak of **{peak_count_rating}** titles in **{peak_year_rating}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"**{top_rating}** leads with **{top_prop:.1f}%** of **{total_titles}** titles, and a **{corr_max:.2f}** correlation exists between **{corr_pair[0]}** and **{corr_pair[1]}**. Ratings average **{movie_mean:.1f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), with a CI of **{movie_stats[0]:.1f}** (CI: **{movie_stats[1][0]:.1f}**-**{movie_stats[1][1]:.1f}**). The histogram peaks at **{movie_hist[0]:.0f}** (**{movie_hist[1]:.1f}%** of titles), and **{movie_prob*100:.1f}%** are TV-14 or lower. Regression shows durations from **{actual_range[0]:.0f}** to **{actual_range[1]:.0f}** minutes with a **{trend_dir}** trend, peaking at **{peak_count_rating}** titles for **{dominant_rating}** in **{peak_year_rating}** and projecting **{pred_ratings[dominant_rating]:,}** by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"**{top_rating}** dominates at **{top_prop:.1f}%**, with a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Ratings average **{movie_stats[0]:.1f}**, with **{movie_prob*100:.1f}%** TV-14 or lower. Regression confirms a **{trend_dir}** trend (average **{mean_actual:.0f}** minutes), projecting **{pred_ratings[dominant_rating]:,}** **{dominant_rating}** titles by **{pred_year}** after a **{peak_count_rating}** peak in **{peak_year_rating}**.")

# ---------- SECTION 6: Top 10 Countries ----------
elif section == "Top 10 Countries":
    st.header("🌍 Top 10 Countries with Most Content")

    # Bar Chart: Top 10 Countries
    st.subheader("📊 Top 10 Countries")
    country_exploded = df.explode('country')
    top_countries = country_exploded['country'].value_counts().head(10)
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax_bar, palette=COLOR_PALETTE)
    ax_bar.set_xlabel("Number of Titles")
    ax_bar.set_ylabel("Country")
    ax_bar.set_title("Top 10 Countries")
    plt.tight_layout()
    st.pyplot(fig_bar)
    top_country = top_countries.idxmax()
    top_count = int(top_countries.max())
    total_count = country_exploded.shape[0]
    top_prop = top_count / total_count * 100
    st.write("*Table: Top 10 Countries*")
    st.dataframe(top_countries.rename("Count").to_frame())
    st.markdown(f"**Insight**: **{top_country}** leads with **{top_count}** titles (**{top_prop:.1f}%** of **{total_count}** titles).")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(country_exploded, numerical_cols, "Top 10 Countries")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Duration)")
    duration_data = country_exploded['duration_num'].dropna()
    movie_duration_data = country_exploded[country_exploded['type'] == 'Movie']['duration_num'].dropna()
    tv_duration_data = country_exploded[country_exploded['type'] == 'TV Show']['duration_num'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_duration_data.describe(),
        'TV Shows': tv_duration_data.describe()
    })
    st.write("*Summary Statistics for Duration*")
    st.dataframe(desc_stats)
    movie_mean = movie_duration_data.mean()
    movie_range = (movie_duration_data.min(), movie_duration_data.max())
    tv_mean = tv_duration_data.mean()
    st.markdown(f"**Insight**: Movie durations average **{movie_mean:.0f} minutes** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), while TV show durations average **{tv_mean:.0f} seasons**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Duration)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_duration_data, "Movies", col1, "minutes"), (tv_duration_data, "TV Shows", col2, "seasons")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_duration_data, "Movies", col1, "minutes", 120), (tv_duration_data, "TV Shows", col2, "seasons", 2)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            lambda_param = data.mean()
            prob = stats.poisson.pmf(threshold, lambda_param)
            tv_prob = prob
    st.markdown(f"""
    **Insight**: The distribution of movie durations shows a peak at approximately **{movie_hist[0]:.0f} minutes**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that duration. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies exceed 120 minutes, indicating a substantial presence of long-form content. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} seasons**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show having exactly **2 seasons** is **{tv_prob*100:.2f}%**, suggesting most shows either stay short or expand significantly.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = country_exploded[country_exploded['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = country_exploded[country_exploded['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    # Debug: Check data shape before regression
    st.write(f"Data shape before regression in Top 10 Countries: {country_exploded.shape}")
    reg_results = regression_analysis(country_exploded, "Top 10 Countries", is_movie=True)
    if reg_results and all(x is not None for x in reg_results):
        trend_dir, r2, rmse, residual_mean = reg_results
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)
    else:
        st.warning("Regression analysis could not be performed due to data issues in Top 10 Countries section.")
        trend_dir = "unknown"

    # Future Trend
    st.subheader("📅 Future Trend")
    top_countries_list = top_countries.index[:3]
    yearly_country_counts = country_exploded[country_exploded['country'].isin(top_countries_list)].groupby(['year_added', 'country']).size().unstack(fill_value=0)
    years = yearly_country_counts.index.astype(int)
    X = years.values.reshape(-1, 1)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    for country in top_countries_list:
        counts = yearly_country_counts.get(country, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_counts = model.predict(future_years)
        future_counts = np.maximum(future_counts, 0)
        ax_trend.plot(years, counts, label=country, marker='o', color=COLOR_PALETTE[list(top_countries_list).index(country) % len(COLOR_PALETTE)])
        ax_trend.plot(future_years.flatten(), future_counts, linestyle='--', color=COLOR_PALETTE[list(top_countries_list).index(country) % len(COLOR_PALETTE)])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Titles by Country")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)

    # Sum of content added from top countries per year
    yearly_totals = yearly_country_counts[top_countries_list].sum(axis=1)
    peak_year_country = int(yearly_totals.idxmax())
    peak_count_country = int(yearly_totals.max())
    dominant_country = yearly_country_counts[top_countries_list].loc[peak_year_country].idxmax()
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    pred_countries = {}
    for country in top_countries_list:
        counts = yearly_country_counts.get(country, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_count = model.predict([[pred_year]])[0]
        future_count = max(future_count, 0)
        pred_countries[country] = int(future_count)
    st.markdown(f"**Insight**: The model projects **{pred_countries[top_country]:,}** **{top_country}** titles by **{pred_year}**, with a peak of **{peak_count_country}** titles in **{peak_year_country}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"**{top_country}** leads with **{top_count}** titles (**{top_prop:.1f}%** of **{total_count}**), and a **{corr_max:.2f}** correlation exists between **{corr_pair[0]}** and **{corr_pair[1]}**. Durations average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), with a CI of **{movie_stats[0]:.0f}** (CI: **{movie_stats[1][0]:.0f}**-**{movie_stats[1][1]:.0f}**). The histogram peaks at **{movie_hist[0]:.0f}** (**{movie_hist[1]:.1f}%** of titles), and **{movie_prob*100:.1f}%** exceed 120 minutes. Regression shows durations from **{actual_range[0]:.0f}** to **{actual_range[1]:.0f}** minutes with a **{trend_dir}** trend, peaking at **{peak_count_country}** titles for **{dominant_country}** in **{peak_year_country}** and projecting **{pred_countries[top_country]:,}** by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"**{top_country}** dominates with **{top_count}** titles (**{top_prop:.1f}%**), and a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Durations average **{movie_stats[0]:.0f}** minutes for movies and **{tv_stats[0]:.0f}** seasons for TV shows, with **{movie_prob*100:.1f}%** of movies exceeding 120 minutes. Regression confirms a **{trend_dir}** trend (average **{mean_actual:.0f}** minutes). Projections estimate **{pred_countries[top_country]:,}** titles from **{top_country}** by **{pred_year}**, after peaking at **{peak_count_country}** in **{peak_year_country}**.")

# ---------- SECTION 7: Most Common Genres ----------
elif section == "Most Common Genres":
    st.header("🎭 Most Common Genres")

    # Bar Chart: Top 10 Genres
    st.subheader("📊 Top 10 Genres")
    genre_exploded = df.explode('listed_in')
    top_genres = genre_exploded['listed_in'].value_counts().head(10)
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax_bar, palette=COLOR_PALETTE)
    ax_bar.set_xlabel("Number of Titles")
    ax_bar.set_ylabel("Genre")
    ax_bar.set_title("Top 10 Genres")
    plt.tight_layout()
    st.pyplot(fig_bar)
    top_genre = top_genres.idxmax()
    top_count = int(top_genres.max())
    total_count = genre_exploded.shape[0]
    top_prop = top_count / total_count * 100
    st.write("*Table: Top 10 Genres*")
    st.dataframe(top_genres.rename("Count").to_frame())
    st.markdown(f"**Insight**: **{top_genre}** leads with **{top_count}** titles (**{top_prop:.1f}%** of **{total_count}** titles).")

    # Correlation Matrix
    numerical_cols = ['duration_num', 'release_year', 'year_added', 'rating_num']
    corr_max, corr_pair = plot_correlation_matrix(genre_exploded, numerical_cols, "Most Common Genres")

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis (Duration)")
    duration_data = genre_exploded['duration_num'].dropna()
    movie_duration_data = genre_exploded[genre_exploded['type'] == 'Movie']['duration_num'].dropna()
    tv_duration_data = genre_exploded[genre_exploded['type'] == 'TV Show']['duration_num'].dropna()
    desc_stats = pd.DataFrame({
        'Movies': movie_duration_data.describe(),
        'TV Shows': tv_duration_data.describe()
    })
    st.write("*Summary Statistics for Duration*")
    st.dataframe(desc_stats)
    movie_mean = movie_duration_data.mean()
    movie_range = (movie_duration_data.min(), movie_duration_data.max())
    tv_mean = tv_duration_data.mean()
    st.markdown(f"**Insight**: Movie durations average **{movie_mean:.0f} minutes** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), while TV show durations average **{tv_mean:.0f} seasons**.")

    # Confidence Intervals
    st.subheader("📏 Confidence Intervals (Duration)")
    col1, col2 = st.columns(2)
    movie_stats = []
    tv_stats = []
    for data, label, col, unit in [(movie_duration_data, "Movies", col1, "minutes"), (tv_duration_data, "TV Shows", col2, "seasons")]:
        desc_stats, mean, ci, ci_text, normality_text = descriptive_stats_and_ci(data, label, unit, bootstrap=(stats.shapiro(data)[1] < 0.05))
        mean_ci = plot_ci_bar(mean, ci, label, unit, col)
        if label == "Movies":
            movie_stats = [mean_ci[0], ci]
        else:
            tv_stats = [mean_ci[0], ci]

    # Probability Distribution
    st.subheader("📈 Probability Distribution")
    col1, col2 = st.columns(2)
    movie_hist = tv_hist = None
    movie_prob = tv_prob = 0
    for data, label, col, unit, threshold in [(movie_duration_data, "Movies", col1, "minutes", 120), (tv_duration_data, "TV Shows", col2, "seasons", 2)]:
        hist_stats = plot_histogram(data, label, unit, col)
        mu, sigma = data.mean(), data.std()
        if label == "Movies":
            movie_hist = hist_stats
            prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
            movie_prob = prob
        else:
            tv_hist = hist_stats
            lambda_param = data.mean()
            prob = stats.poisson.pmf(threshold, lambda_param)
            tv_prob = prob
    st.markdown(f"""
    **Insight**: The distribution of movie durations shows a peak at approximately **{movie_hist[0]:.0f} minutes**, 
    with around **{movie_hist[1]:.1f}%** of movies centered around that duration. Additionally, approximately 
    **{movie_prob*100:.1f}%** of movies exceed 120 minutes, indicating a substantial presence of long-form content. 
    
    For TV shows, the peak is at **{tv_hist[0]:.0f} seasons**, with about **{tv_hist[1]:.1f}%** clustered around this value. 
    The probability of a show having exactly **2 seasons** is **{tv_prob*100:.2f}%**, suggesting most shows either stay short or expand significantly.
    """)

    # Conditional Probability
    st.subheader("🎲 Conditional Probability")
    st.markdown("**Top 3 Ratings by Content Type**")
    movie_rating_probs = genre_exploded[genre_exploded['type'] == 'Movie']['rating'].value_counts(normalize=True).head(3) * 100
    tv_rating_probs = genre_exploded[genre_exploded['type'] == 'TV Show']['rating'].value_counts(normalize=True).head(3) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎬 **Movies**")
        st.dataframe(movie_rating_probs.rename("Probability (%)").to_frame())
    with col2:
        st.write("📺 **TV Shows**")
        st.dataframe(tv_rating_probs.rename("Probability (%)").to_frame())
    st.markdown(f"""
    **Insight**: Among movies, the most common rating is **{movie_rating_probs.idxmax()}** with a probability of **{movie_rating_probs.max():.2f}%**. 
    For TV shows, **{tv_rating_probs.idxmax()}** is the most frequent, with **{tv_rating_probs.max():.2f}%**. This reflects rating trends and target audiences.
    """)

    # Regression Analysis
    st.subheader("📈 Regression Analysis")
    reg_results = regression_analysis(genre_exploded, "Most Common Genres", is_movie=True)
    if reg_results and all(x is not None for x in reg_results):
        trend_dir, r2, rmse, residual_mean = reg_results
        st.markdown(f"""
        **Insight**: The regression model indicates a **{trend_dir}** relationship between release year and content duration, 
        with an R² of **{float(r2):.4f}** and RMSE of **{float(rmse):.2f} minutes**. 
        
        - **R² (coefficient of determination)** shows that **{float(r2)*100:.2f}%** of the variation in duration can be explained by the release year and rating.
        - **RMSE** quantifies the model’s prediction error, with average deviation of **{float(rmse):.2f} minutes** from actual values.

        The scatterplot further suggests that while there is a trend, there’s notable variance in durations especially for newer releases, 
        implying that release year isn't the sole factor influencing content length.
        """)
    else:
        st.warning("Regression analysis could not be performed due to data issues in Most Common Genres section.")
        trend_dir = "unknown"

    # Future Trend
    st.subheader("📅 Future Trend")
    top_genres_list = top_genres.index[:3]
    yearly_genre_counts = genre_exploded[genre_exploded['listed_in'].isin(top_genres_list)].groupby(['year_added', 'listed_in']).size().unstack(fill_value=0)
    years = yearly_genre_counts.index.astype(int)
    X = years.values.reshape(-1, 1)
    future_years = np.arange(max(years) + 1, 2031).reshape(-1, 1)
    fig_trend, ax_trend = plt.subplots(figsize=(8, 6))
    for genre in top_genres_list:
        counts = yearly_genre_counts.get(genre, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_counts = model.predict(future_years)
        future_counts = np.maximum(future_counts, 0)
        ax_trend.plot(years, counts, label=genre, marker='o', color=COLOR_PALETTE[list(top_genres_list).index(genre) % len(COLOR_PALETTE)])
        ax_trend.plot(future_years.flatten(), future_counts, linestyle='--', color=COLOR_PALETTE[list(top_genres_list).index(genre) % len(COLOR_PALETTE)])
    ax_trend.set_xlabel("Year")
    ax_trend.set_ylabel("Number of Titles")
    ax_trend.set_title("Historical and Projected Titles by Genre")
    ax_trend.legend()
    plt.tight_layout()
    st.pyplot(fig_trend)

    # Sum of content added from top genres per year
    yearly_totals = yearly_genre_counts[top_genres_list].sum(axis=1)
    peak_year_genre = int(yearly_totals.idxmax())
    peak_count_genre = int(yearly_totals.max())
    dominant_genre = yearly_genre_counts[top_genres_list].loc[peak_year_genre].idxmax()
    pred_year = st.slider("Select Year for Prediction", int(max(years)) + 1, 2030, int(max(years)) + 1)
    pred_genres = {}
    for genre in top_genres_list:
        counts = yearly_genre_counts.get(genre, pd.Series(0, index=years))
        model = LinearRegression().fit(X, counts)
        future_count = model.predict([[pred_year]])[0]
        future_count = max(future_count, 0)
        pred_genres[genre] = int(future_count)
    st.markdown(f"**Insight**: The model projects **{pred_genres[top_genre]:,}** **{top_genre}** titles by **{pred_year}**, with a peak of **{peak_count_genre}** titles in **{peak_year_genre}**.")

    # Consolidated Description
    st.subheader("📝 Description")
    actual_range = (df['duration_num'].min(), df['duration_num'].max())
    st.markdown(f"**{top_genre}** leads with **{top_count}** titles (**{top_prop:.1f}%** of **{total_count}**), and a **{corr_max:.2f}** correlation exists between **{corr_pair[0]}** and **{corr_pair[1]}**. Durations average **{movie_mean:.0f}** (range: **{movie_range[0]:.0f}**-**{movie_range[1]:.0f}**), with a CI of **{movie_stats[0]:.0f}** (CI: **{movie_stats[1][0]:.0f}**-**{movie_stats[1][1]:.0f}**). The histogram peaks at **{movie_hist[0]:.0f}** (**{movie_hist[1]:.1f}%** of titles), and **{movie_prob*100:.1f}%** exceed 120 minutes. Regression shows durations from **{actual_range[0]:.0f}** to **{actual_range[1]:.0f}** minutes with a **{trend_dir}** trend, peaking at **{peak_count_genre}** titles for **{dominant_genre}** in **{peak_year_genre}** and projecting **{pred_genres[top_genre]:,}** by {pred_year}.")

    # Consolidated Conclusion
    st.subheader("✅ Conclusion")
    mean_actual = df['duration_num'].mean()
    st.markdown(f"**{top_genre}** dominates with **{top_count}** titles (**{top_prop:.1f}%**), and a **{corr_max:.2f}** correlation between **{corr_pair[0]}** and **{corr_pair[1]}**. Durations average **{movie_stats[0]:.0f}** minutes for movies and **{tv_stats[0]:.0f}** seasons for TV shows, with **{movie_prob*100:.1f}%** of movies exceeding 120 minutes. Regression confirms a **{trend_dir}** trend (average **{mean_actual:.0f}** minutes). Projections estimate **{pred_genres[top_genre]:,}** **{top_genre}** titles by **{pred_year}**, after peaking at **{peak_count_genre}** in **{peak_year_genre}**.")
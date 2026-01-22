# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

# -----------------------------------------------

# Page config
st.set_page_config(
    page_title="Amazon Job Reviews EDA",
    layout="wide"
)




# ------------------------------------
# Page config
# ------------------------------------
st.set_page_config(
    page_title="Amazon Job Reviews EDA",
    layout="wide"
)


st.title("Amazon Job Reviews EDA (2008–2020)")
st.markdown("Interactive exploratory analysis of employee reviews across countries ")

# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_job_reviews_country_year (1).csv")
    df.columns = df.columns.str.strip()  # remove hidden spaces
    return df

df = load_data()

# Ensure Country column exists
if "Country" not in df.columns:
    if "Location" in df.columns:
        df = df.rename(columns={"Location": "Country"})
    else:
        st.error("Neither 'Country' nor 'Location' column found in dataset.")
        st.stop()



# Identify numeric metrics automatically
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["Year", "ID number"]]


# Text column
text_col = [c for c in df.columns if "review" in c.lower() or "text" in c.lower()][0]

# ------------------------------------
# Sidebar controls
# ------------------------------------
st.sidebar.header("Controls")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

countries = st.sidebar.multiselect(
    "Select Countries",
    options=df["Country"].unique(),
    default=list(df["Country"].unique())
)

selected_metrics = st.sidebar.multiselect(
    "Select Rating Metrics",
    options=numeric_cols,
    default=numeric_cols
)

filtered_df = df[
    (df["Year"].between(year_range[0], year_range[1])) &
    (df["Country"].isin(countries))
]

wordcloud_insights = {
    ("USA", "Pros"):
        "Positive reviews from the USA frequently emphasize pay, benefits,work environment and team. \n This shows a general appreciation of the internal work culture and the financial compensation at Amazon",

    ("USA", "Cons"):
        "Negative feedback from the USA commonly highlights words like work, rime, people and manager showing concerns about work-life balance, work intensity and management issues which can be areas of potential growth.",

    ("USA", "Advice to Management"):
        "Advice from US employees often focuses on words like manager,time,team and management showing there exists a need for improving leadership communication and sustaining employee well-being.",

    ("India", "Pros"):
        "Indian employees frequently highlight pay,work and benefits suggesting Indian employees mostly agree with their US counterparts regarding the strongpoints of being employed at Amazon ",

    ("India", "Cons"):
        "Concerns from Indian reviews shows words like work, time,hour break and long suggesting the cons often center around demanding work culture and long hourse.",

    ("India", "Advice to Management"):
        "Advice from Indian employees includes words like better, manager, time and management which suggests a demand for improving people management, workload distribution, and overall team support."
}

# ------------------------------------
# Tabs for navigation
# ------------------------------------
tabs = st.tabs([
    "Home",
    "Data Description",
    "Yearly Averages Table",
    "Correlation Heatmap",
    "Country-wise Trends",
    "Multivariable Trends",
    "Word Clouds",
    "Categorical Insights",
    "Overall Conclusions"
])

metric_conclusions = {
    "Overall Rating": "Overall ratings are higher in the USA, while India shows more variability.",
    "Work-Life Balance": "Work-life balance ratings are more tightly clustered in the USA.",
    "Compensation & Benefits": "Compensation ratings are generally higher in the USA with fewer low outliers.",
    "Career Opportunities": "Both countries show similar medians, but India has wider dispersion.",
    "Culture & Values": "Cultural ratings are balanced, with fewer extreme lows in the USA.",
    "Senior Management": "Management ratings show greater polarization in India."
}

# ------------------------------------
# 0. Home Tab
# ------------------------------------
with tabs[0]:
    st.subheader("Welcome to my Amazon Job Reviews EDA Dashboard!")
    st.markdown(
        """
        This interactive dashboard allows you to explore employee reviews of Amazon from 2008 to 2020 across different countries.
        
        Our dataset includes various job satisfaction metrics such as Overall Rating, Work-Life Balance, Compensation & Benefits, Career Opportunities, Culture & Values, and Senior Management.

        Use the sidebar to filter data by year range, countries, and specific rating metrics. Navigate through the tabs to view different visualizations and analyses.

        The Dropdown allows you to select specific metrics you wish to visualize which will be useful for analysis across a few specific metrics 
            """
    )
# ------------------------------------
# ------------------------------------
# 0.5. Data Description
# ------------------------------------
with tabs[1]:
    st.subheader("Dataset Overview")
    st.markdown(
        """
        **Data Source:** The dataset is sourced from publicly available employee reviews gathered from Glassdoor

        The dataset contains employee reviews of Amazon from 2008 to 2020 across multiple countries. Each review includes various job satisfaction metrics rated on a scale, along with written feedback in the form of pros, cons, and advice to management.
        
        The following are the main columns/metrics in the dataset:
        """
    )
    st.markdown("""
### Column Descriptions

- **ID number (Integer):** Unique identifier for each review.  
- **Date (Character):** Date of the review (day–month–year format).  
- **Location (Character):** Job location (city/state/country).  
- **Position (Character):** Employee’s job title/role.  
- **Comment for company (Character):** Overall textual comment summarizing the review.  
- **Overall rating (Numeric):** Overall satisfaction rating (1–5).  
- **Work/Life Balance (Numeric):** Rating for work–life balance (1–5).  
- **Culture & Values (Numeric):** Rating for company culture and values (1–5).  
- **Diversity & Inclusion (Numeric):** Rating for diversity and inclusion (1–5, limited data available).  
- **Career Opportunities (Numeric):** Rating for growth and career opportunities (1–5).  
- **Compensation and Benefits (Numeric):** Rating for pay and benefits (1–5).  
- **Senior Management (Numeric):** Rating for management quality (1–5).  
- **CEO Approval (Character):** Whether employees approve of the CEO (yes, no, may be).  
- **Recommended (Character):** Whether the reviewer recommends Amazon as a workplace.  
- **Business Outlook (Character):** Reviewer’s perception of the company’s future (positive, negative, neutral).  
- **Current employee (Boolean):** Whether the reviewer is a current employee.  
- **Former employee (Boolean):** Whether the reviewer is a former employee.  
- **Timeline (Character):** Employment timeline (tenure period where available).  
- **cons (Character):** Reported disadvantages of working at Amazon.  
- **pros (Character):** Reported advantages of working at Amazon.  
- **advice to Management (Character):** Suggestions for company leadership.  
- **review_url (Character):** Link to the original Glassdoor review.
                
Our Data is exclusively from employees in the USA and India.
                
Below you can find an interactive pie chart showing the distribution of reviews by country across the entire dataset:
""")

    st.subheader("Distribution of Reviews by Country")

    country_counts = (
        filtered_df
        .groupby("Country")
        .size()
        .reset_index(name="Number of Reviews")
    )

    fig = px.pie(
        country_counts,
        names="Country",
        values="Number of Reviews",
        hole=0.4,
        title="Share of Reviews by Country"
    )

    fig.update_traces(textinfo="percent+label")

    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "This pie chart shows how employee reviews are distributed across countries "
        "for the selected year range and country filters."
    )
st.subheader("Calendar Heatmap")

st.image(
    "calendar_heatmap.png",
    caption="Calendar Heatmap of Review Activity",
    use_container_width=True
)

# ------------------------------------


# 1. Yearly averages table
# ------------------------------------
with tabs[2]:
    st.subheader("Year-by-Year Average Metrics")

    table = (
        filtered_df
        .groupby("Year")[selected_metrics]
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(table, use_container_width=True)

    st.info("Summarizes annual trends numerically.")
    st.info(
        """
        ***Key Insights:***

The year-wise averages (2008–2020) reveal several important trends in Amazon employee reviews:

• Overall Rating increased steadily from ~3.25 in 2008 to ~3.7 by 2020. This reflects a long-term
improvement in employee sentiment, despite short-term dips during Amazon’s rapid expansion years.

• Work–Life Balance consistently lagged behind other metrics. It declined below 3.0 between 2010–
2015 (lowest in 2013), highlighting the intensity of Amazon’s work culture during its high-growth phase.
Although it recovered slightly in later years, it remained the weakest dimension overall.

• Career Opportunities and Compensation & Benefits showed strong upward trends, especially
after 2012. By 2020 both exceeded 3.8, suggesting that Amazon’s rapid growth, market dominance,
and pay improvements boosted employee perceptions of growth potential and rewards.

• Senior Management dipped in the mid-2010s (2013–2015), coinciding with public criticism of Amazon’s demanding workplace culture (e.g., the 2015 New York Times article). Ratings improved afterwards, indicating gradual adaptation in leadership and communication practices.

Employee sentiment at Amazon became more positive over the 12-year span. Compensation and career
growth opportunities emerged as the strongest drivers of improvement, while work–life balance and
management quality remained areas of concern. The data portrays Amazon as a workplace offering
excellent financial and professional incentives, but often at the cost of personal time and wellbeing.
We also lack any data that rates culture and values before 2012 which shows that the metric was not taken
into consideration pre-2012 as well as lacking all data regarding"""
    )
# ------------------------------------

# 2. Correlation heatmap
# ------------------------------------
with tabs[3]:
    st.subheader("Correlation Between Rating Metrics")

    corr = filtered_df[selected_metrics].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("Highlights relationships between different job satisfaction metrics.")
    st.info(
    """
    **Key Insights from the Correlation Analysis**

    • Overall Rating shows strong positive relationships with all other metrics, indicating that employees’ overall satisfaction reflects multiple aspects of their work experience.

    • Senior Management and Work–Life Balance are closely linked, suggesting that effective leadership is associated with better work–life outcomes.

    • Compensation & Benefits and Career Opportunities are important contributors to overall satisfaction, highlighting the role of tangible rewards and growth prospects.

    • While most metrics move together, leadership quality and work–life balance appear especially influential in shaping employees’ broader perception of the company.
    """
)


# ------------------------------------
# ------------------------------------
# Country-wise Trends (Boxplot + Line)
# ------------------------------------
with tabs[4]:
    st.subheader("Country-wise Rating Trends")

    metric = st.selectbox(
        "Select Rating Metric",
        selected_metrics,
        key="country_trend_metric"
    )

    # ---------- Boxplot ----------
    st.markdown("#### Distribution of Ratings by Country")

    fig_box = px.box(
        filtered_df,
        x="Country",
        y=metric,
        color="Country"
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        metric_conclusions.get(
            metric,
            "Compares rating distributions across countries."
        )
    )

    st.divider()

    # ---------- Line plot ----------
    st.markdown("#### Trends Over Time by Country")

    yearly = (
        filtered_df
        .groupby(["Year", "Country"])[metric]
        .mean()
        .reset_index()
    )

    fig_line = px.line(
        yearly,
        x="Year",
        y=metric,
        color="Country",
        markers=True
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.info(
        "Shows how the selected metric evolves over time for each country "
        "under the current filters."
    )

    st.info(

    """
    **Key Takeaways**

    **US reviews** appear more **stable** across metrics, while **Indian reviews** show **greater variability**.

    By the late 2010s, ratings across countries converge, potentially reflecting improvements in global HR practices and evolving workplace conditions.
    
    **More general trends in every metric on an individual basis are as follows:**
    
    • **Overall Satisfaction:** Ratings in both India and the USA improve steadily over time and converge by 2020, though India shows greater variability in individual experiences.

    • **Career Opportunities:** Both countries rate career growth positively. Trends are steadier in the USA, while India exhibits more fluctuation before recovering in later years.

    • **Compensation & Benefits:** Ratings trend upward in both regions, with very similar central tendencies, indicating broadly comparable perceptions of compensation.

    • **Culture & Values:** Cultural alignment is rated more favorably in the USA, with both countries showing temporary declines in the early 2010s followed by recovery.

    • **Senior Management:** Leadership is consistently among the lower-rated dimensions in both countries, though perceptions improve modestly after the mid-2010s.

    • **Work–Life Balance:** This remains the weakest-rated metric across regions, with only gradual improvement in recent years and substantial variability throughout.
    """
    
)

# ------------------------------------

# 5. Multivariable line plots
# ------------------------------------
with tabs[5]:
    st.subheader("Multivariable Trends Over Time")

    yearly_multi = (
        filtered_df
        .groupby("Year")[selected_metrics]
        .mean()
        .reset_index()
        .melt(id_vars="Year", var_name="Metric", value_name="Average")
    )

    fig = px.line(
        yearly_multi,
        x="Year",
        y="Average",
        color="Metric",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Allows comparison of all numeric metrics simultaneously.")

    st.info(
    """
    **Year-wise Averages: Key Insights (2008–2020)**

    • **Overall Rating:** Shows a steady long-term increase, indicating gradual improvement in employee sentiment despite short-term fluctuations during expansion phases.

    • **Work–Life Balance:** Declines in the early 2010s before stabilizing, suggesting sustained pressure during Amazon’s high-growth period with limited recovery.

    • **Career Opportunities & Compensation:** Both metrics improve markedly after 2012, reflecting stronger perceptions of growth opportunities and financial incentives.

    • **Senior Management:** Experiences a mid-2010s decline followed by recovery, aligning with periods of public scrutiny and subsequent organizational adjustments.

    Overall, compensation and career growth emerge as the strongest areas of improvement, while work–life balance and leadership remain persistent concerns.
    """
)
    st.info(
    """
    **Impact of COVID-19 on Employee Sentiment (Pre- vs Post-2019)**

    • **Overall Rating:** Increases during the pandemic, suggesting slightly more positive overall perceptions despite challenging conditions.

    • **Work–Life Balance:** Improves modestly, likely influenced by remote or hybrid work arrangements and increased flexibility.

    • **Career Opportunities:** Shows noticeable improvement, potentially driven by rapid expansion in logistics, cloud services, and related sectors.

    • **Compensation & Benefits:** Trends upward, reflecting pay raises, bonuses, and additional benefits introduced during the pandemic.

    • **Senior Management:** Improves relative to pre-pandemic years, indicating greater approval of leadership decisions during crisis management.

    Collectively, the COVID-19 period does not appear to negatively impact internal employee sentiment. Instead, ratings suggest a neutral to mildly positive effect on overall satisfaction.
    """
)


# ------------------------------------
# 6. Word clouds
# ------------------------------------
with tabs[6]:
    st.subheader("Word Clouds from Written Reviews")

    # Country selector
    country_wc = st.selectbox(
        "Select Country",
        options=countries,
        key="wc_country"
    )

    # Word cloud type selector
    wc_type = st.selectbox(
        "Select Review Type",
        options=["Pros", "Cons", "Advice to Management"],
        key="wc_type"
    )

    # Map dropdown label → actual column name
    wc_column_map = {
        "Pros": "pros",
        "Cons": "cons",
        "Advice to Management": "advice to Management"
    }

    text_col = wc_column_map[wc_type]

    # Build text corpus
    text = " ".join(
        filtered_df[
            filtered_df["Country"] == country_wc
        ][text_col]
        .dropna()
        .astype(str)
    )

    # Handle empty text safely
    if text.strip() == "":
        st.warning("No text available for the selected filters.")
    else:
        wc = WordCloud(
            stopwords=STOPWORDS,
            background_color="white",
            width=800,
            height=400
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    # Dynamic insight
    st.info(
        wordcloud_insights.get(
            (country_wc, wc_type),
            "Displays commonly used words in employee reviews for the selected filters."
        )
    )


# ------------------------------------

# ------------------------------------
# 7. Categorical Insights
# ------------------------------------
with tabs[7]:
    st.subheader("Categorical Insights: Employee Sentiment")

    st.markdown(
        """
        This section analyzes categorical responses related to employee sentiment,
        such as **CEO Approval** and **Recommendation of Amazon as a workplace**.
        All results reflect the selected year range and country filters.
        """
    )

    # Select categorical column
    cat_col = st.selectbox(
        "Select Categorical Variable",
        options=["CEO Approval", "Recommended"]
    )

    # Clean + count
    cat_counts = (
        filtered_df[cat_col]
        .value_counts(dropna=True)
        .reset_index()
    )
    cat_counts.columns = [cat_col, "Count"]

    # Percentage calculation
    cat_counts["Percentage"] = (
        cat_counts["Count"] / cat_counts["Count"].sum() * 100
    ).round(2)

    # Bar chart (preferred for categorical data)
    fig = px.bar(
        cat_counts,
        x=cat_col,
        y="Count",
        text="Percentage",
        title=f"Distribution of {cat_col}"
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_title="Number of Reviews")

    st.plotly_chart(fig, use_container_width=True)

    # Display table
    st.markdown("#### Summary Table")
    st.dataframe(cat_counts, use_container_width=True)

    st.info(
        "These categorical distributions highlight employee sentiment without imposing "
        "numeric assumptions on qualitative responses."
    )

# ------------------------------------
# Overall Conclusions
# ------------------------------------
with tabs[8]:
    st.subheader("Overall Conclusions & Key Takeaways")

    st.markdown(
        """
        This section summarizes the key insights drawn from employee reviews of Amazon
        across time periods and regions, combining quantitative ratings and qualitative feedback.
        """
    )

    st.info(
        """
        **Key Findings**

        • Employee sentiment at Amazon is shaped by both **temporal changes** and **regional context**. 
        Ratings generally improve over time, with noticeable dips during high-growth phases and recovery in later years.

        • **Compensation and Career Opportunities** emerge as Amazon’s strongest aspects globally, showing consistent improvement
        and contributing positively to overall satisfaction.

        • **Work–Life Balance and Senior Management** remain persistent areas of concern across regions, despite partial improvements after 2016.

        • Reviews from the **USA** tend to be more stable and consistent, particularly in compensation and culture,
        while **Indian reviews** exhibit greater variability, reflecting more diverse employee experiences.

        • Qualitative feedback reinforces these patterns, highlighting workload intensity, leadership challenges,
        and work–life balance as recurring themes, alongside appreciation for growth opportunities and pay.

        • Overall, Amazon is perceived as a **career accelerator** that offers strong professional and financial rewards,
        but sustaining employee satisfaction over time will require continued attention to workload management
        and leadership quality.
        """
    )

    st.caption(
        "These conclusions are based on aggregated trends from 2008–2020 and should be interpreted in the context "
        "of review volume, regional differences, and evolving organizational practices."
    )


# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")

st.markdown("Built with Streamlit •  by Agnivesh Chatterjee ")

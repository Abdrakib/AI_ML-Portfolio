# Customer Segmentation with Machine Learning

Comprehensive customer segmentation project using multiple clustering algorithms to identify distinct customer groups for targeted marketing strategies.

## Description

This project implements a complete customer segmentation pipeline using unsupervised machine learning techniques. It segments customers into distinct groups based on their demographics, income, and spending behavior. The segmentation enables businesses to develop targeted marketing strategies, improve customer retention, and optimize resource allocation.

## Problem Definition

Given customer data (demographics, income, spending behavior), can we identify distinct customer segments that share similar characteristics and behaviors?

## Pipeline

### 1. Data Collection / Loading
- Load Mall Customers dataset from CSV file
- Dataset contains:
  - **CustomerID**: Unique customer identifier
  - **Gender**: Customer gender (Male, Female)
  - **Age**: Customer age
  - **Annual Income (k$)**: Annual income in thousands of dollars
  - **Spending Score (1-100)**: Spending score assigned by the mall (1-100 scale)

### 2. Data Cleaning
- **Check for Duplicates**: Identify and remove duplicate customer records
- **Handle Missing Values**: 
  - Fill numerical missing values with median
  - Fill categorical missing values with mode
- **Detect and Handle Outliers**: 
  - Use IQR (Interquartile Range) method to detect outliers
  - Cap outliers instead of removing (preserve data)
  - Apply to Age, Annual Income, and Spending Score

### 3. Feature Engineering ⭐

Critical step that creates meaningful features from available data:

- **RFM-like Features** (using available data):
  - **Monetary**: Annual Income converted to actual dollars (represents spending capacity)
  - **Frequency**: Spending Score (represents engagement/frequency of visits)
  - **Recency**: Inverse of Spending Score (lower score might indicate less recent activity)

- **Age Groups**: Categorize customers into groups (Young, Adult, Middle-aged, Senior, Elderly)

- **Income-to-Spending Ratio**: 
  - Higher ratio = high income but low spending (potential for upselling)
  - Lower ratio = spending close to income capacity (loyal customers)

- **Customer Value Score**: 
  - Normalized combination of income and spending behavior
  - Weighted average: 50% income normalized + 50% spending normalized

- **Spending Categories**: Low, Medium, High, Very High (based on Spending Score)

- **Income Categories**: Low, Medium, High, Very High (based on Annual Income)

- **Activity Score**: Normalized Spending Score (0-1 scale) as proxy for engagement

### 4. Encode + Scale Data

**Critical for distance-based clustering algorithms**

- **One-Hot Encoding**: Encode categorical variables (Gender)
  - Fit encoder on training data only
  - Transform all data using same encoding
  
- **Feature Scaling**: StandardScaler for all numerical features
  - Standardize features to have mean=0 and std=1
  - Essential because clustering is distance-based
  - All features must be on similar scale

### 5. Choose Algorithm

Multiple clustering algorithms tested to find the best fit:

- **K-Means**: 
  - Fast, works well with spherical clusters
  - Requires number of clusters (k) to be specified
  - Most commonly used for customer segmentation

- **Hierarchical Clustering**: 
  - Good for hierarchical relationships
  - Creates dendrogram showing cluster relationships
  - Uses ward linkage

- **DBSCAN**: 
  - Handles noise and non-spherical clusters
  - Doesn't require number of clusters
  - Can identify outliers as noise points

- **Gaussian Mixture Model (GMM)**: 
  - Probabilistic approach
  - Handles overlapping clusters
  - Provides probability of cluster membership

### 6. Pick Number of Clusters

Determine optimal number of clusters using multiple methods:

- **Elbow Method**: 
  - Plot inertia (within-cluster sum of squares) vs number of clusters
  - Find the "elbow" point where inertia decreases more slowly
  - Visual method for selecting k

- **Silhouette Score**: 
  - Measure cluster separation and cohesion
  - Range: -1 to 1 (higher is better)
  - Calculate for k values from 2 to 10
  - Select k with highest silhouette score

- **Silhouette Analysis**: 
  - Detailed visualization for different k values
  - Shows silhouette coefficient for each sample
  - Helps understand cluster quality and separation

### 7. Train Clustering Models

Fit models on full dataset (unsupervised learning - no labels):

- **K-Means**: Train with optimal k determined from analysis
- **Hierarchical Clustering**: Train with optimal k
- **DBSCAN**: Train with eps and min_samples parameters
- **GMM**: Train with optimal number of components

All models trained and cluster assignments stored for comparison.

### 8. Evaluate Clusters

Comprehensive evaluation of all clustering algorithms:

- **Silhouette Score**: 
  - Calculate for each algorithm
  - Compare across all models
  - Higher score = better cluster separation

- **Cluster Separation**: 
  - Visual analysis of cluster boundaries
  - Statistical analysis of cluster characteristics

- **Model Comparison**: 
  - Compare all algorithms side-by-side
  - Select best model based on silhouette score
  - Consider interpretability and business needs

### 9. Visualize

Comprehensive visualizations to understand clusters:

- **PCA Plots**: 
  - 2D visualization of clusters in reduced space
  - Shows cluster centers and boundaries
  - Displays explained variance ratio

- **Key Feature Scatter Plots**: 
  - Annual Income vs Spending Score
  - Age vs Spending Score
  - Annual Income vs Age
  - Color-coded by cluster

- **Boxplots**: 
  - Feature distribution by cluster
  - Shows median, quartiles, and outliers
  - Helps understand cluster characteristics

- **Cluster Size Distribution**: 
  - Bar charts showing number of customers per cluster
  - Helps identify segment sizes

### 10. Profile Segments ⭐⭐

Translate clusters into human-readable business insights:

- **Calculate Average Metrics**: 
  - Average Age, Annual Income, Spending Score per cluster
  - Average Customer Value Score, Activity Score
  - Most common Gender and Age Group per cluster

- **Identify Segment Characteristics**: 
  - High-Value High-Spending Customers
  - High-Income Low-Spending Customers (upsell opportunity)
  - High-Spending Moderate-Income Customers
  - Low-Value Customers
  - Moderate Customers

- **Create Descriptive Labels**: 
  - Human-readable segment descriptions
  - Age-based categorization (Older/Younger)
  - Spending behavior classification

### 11. Business Recommendations

Actionable insights and strategies for each customer segment:

- **High-Value High-Spending Customers**:
  - VIP Rewards Program: Offer exclusive benefits and early access
  - Premium Products: Showcase high-end merchandise
  - Personal Shopping Assistant: Dedicated support
  - Loyalty Program: Exclusive rewards for frequent purchases

- **High-Income Low-Spending Customers**:
  - Upsell Campaign: Promote premium products and services
  - Personalized Recommendations: Show products matching their income level
  - Exclusive Events: Invite to VIP shopping events
  - Bundle Offers: Encourage higher spending

- **High-Spending Moderate-Income Customers**:
  - Value Deals: Offer discounts and promotions
  - Loyalty Rewards: Reward frequent shopping
  - Budget-Friendly Options: Highlight affordable products
  - Payment Plans: Offer flexible payment options

- **Low-Value Customers**:
  - Engagement Campaign: Increase mall visits
  - Promotional Offers: Attract with discounts
  - New Customer Welcome: Special offers for first-time shoppers
  - Feedback Collection: Understand their needs

- **Standard Customers**:
  - Regular Marketing: Newsletters and promotions
  - Seasonal Campaigns: Holiday specials and events
  - Product Recommendations: Based on similar customers
  - Cross-Promotions: Partner with other stores

## Models

- **Primary Model**: K-Means Clustering (selected based on silhouette score)
- **Alternative Models**: Hierarchical Clustering, DBSCAN, Gaussian Mixture Model (GMM)
- **Optimal Clusters**: Determined using Elbow Method and Silhouette Analysis (typically 3-6 clusters)

## Evaluation Metrics

- **Primary Metric**: **Silhouette Score** (measures cluster separation and cohesion)
  - Range: -1 to 1
  - Higher is better
  - Values > 0.5 indicate good clustering

- **Secondary Metrics**: 
  - **Inertia** (for K-Means): Within-cluster sum of squares (lower is better)
  - **Cluster Size Distribution**: Balance of cluster sizes
  - **Feature Statistics per Cluster**: Mean, median, std for each feature

## Features Used

### Original Features
- **CustomerID**: Unique identifier
- **Gender**: Male, Female
- **Age**: Customer age in years
- **Annual Income (k$)**: Annual income in thousands
- **Spending Score (1-100)**: Spending behavior score

### Engineered Features
- **Monetary**: Annual Income × 1000 (actual dollars)
- **Frequency**: Spending Score (engagement proxy)
- **Recency**: 101 - Spending Score (inverted for recency)
- **Age Group**: Categorical (Young, Adult, Middle-aged, Senior, Elderly)
- **Income-to-Spending Ratio**: Income / (Spending Score + 1)
- **Customer Value Score**: Normalized combination of income and spending
- **Spending Category**: Low, Medium, High, Very High
- **Income Category**: Low, Medium, High, Very High
- **Activity Score**: Normalized Spending Score (0-1)

## Output

The project generates:
1. **customer_segments.csv**: Complete dataset with cluster assignments and segment descriptions
2. **Visualizations**: 
   - PCA plots showing cluster separation
   - Scatter plots of key features by cluster
   - Boxplots showing feature distributions
   - Cluster size distribution charts
3. **Cluster Profiles**: Statistical summary table of each segment
4. **Business Recommendations**: Actionable strategies for each customer segment
5. **Model Comparison**: Evaluation metrics for all clustering algorithms

## Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

Optional (for interactive visualizations):
```txt
plotly>=5.0.0
```

## Applications

- **Marketing Strategies**: Targeted campaigns for each segment
- **Customer Retention**: Identify and re-engage at-risk customers
- **Resource Allocation**: Focus efforts on high-value segments
- **Product Development**: Understand customer needs by segment
- **Pricing Strategies**: Segment-based pricing optimization
- **Customer Service**: Prioritize support for VIP customers
- **Inventory Management**: Stock products based on segment preferences
- **Store Layout**: Design store sections based on customer segments

## Key Insights

1. **Feature Engineering is Critical**: Creating meaningful features (RFM-like, value scores) significantly impacts clustering quality
2. **Scaling is Essential**: Distance-based algorithms require all features to be on similar scales
3. **Multiple Algorithms Should be Tested**: Different algorithms may reveal different patterns
4. **Silhouette Score is Key**: Best metric for evaluating cluster quality in unsupervised learning
5. **Business Context Matters**: Translating clusters into actionable insights requires domain knowledge
6. **Optimal k Varies**: Number of clusters depends on business needs and data characteristics
7. **Visualization Helps**: PCA and scatter plots make clusters interpretable

## Usage

1. **Load Data**: Update the file path in the notebook to point to your `Mall_Customers.csv` file
2. **Run All Cells**: Execute all cells sequentially
3. **Review Results**: 
   - Check optimal number of clusters
   - Review cluster profiles
   - Analyze business recommendations
4. **Customize**: 
   - Adjust feature engineering based on your data
   - Modify business recommendations for your industry
   - Change number of clusters if needed

## Data Format

The notebook expects a CSV file with the following columns:
- **CustomerID**: Unique customer identifier (integer)
- **Gender**: Customer gender (string: Male/Female)
- **Age**: Customer age (integer)
- **Annual Income (k$)**: Annual income in thousands (integer)
- **Spending Score (1-100)**: Spending score (integer, 1-100)

Column names can have spaces and special characters - the notebook handles them automatically.

## Notes

- **Data Source**: Uses Mall_Customers.csv dataset (200 customers, 5 features)
- **Scalability**: Methods work with larger datasets, but computation time increases
- **Interpretability**: K-Means and Hierarchical are more interpretable than DBSCAN or GMM
- **Business Alignment**: Segment descriptions should align with business goals
- **Regular Updates**: Re-segment customers periodically as behavior changes
- **Feature Selection**: Consider removing low-importance features to improve clustering

## Model Performance Summary

After evaluation:
- **Best Model**: Selected based on highest silhouette score
- **Cluster Quality**: Silhouette scores typically range from 0.3 to 0.6 for customer data
- **Segment Balance**: Clusters should have reasonable size distribution (not too imbalanced)
- **Practical Use**: Models effectively identify distinct customer groups for targeted marketing

## Recommendations

1. **Use Best Model**: Deploy the model with highest silhouette score
2. **Monitor Segments**: Track how customers move between segments over time
3. **A/B Testing**: Test different marketing strategies on each segment
4. **Regular Retraining**: Re-cluster customers quarterly or when significant changes occur
5. **Combine with Other Data**: Integrate with transaction history, website behavior, etc. for richer segmentation

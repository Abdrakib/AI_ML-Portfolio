# Market Basket Analysis

Comprehensive market basket analysis project to discover product associations and patterns in retail transactions using association rule mining.

## Description

This project performs market basket analysis on retail transaction data to identify:
- Products frequently bought together
- Association rules with high confidence and lift
- Product co-occurrence patterns
- Insights for cross-selling and product recommendations

## Pipeline

1. **Data Loading** - Load transaction data from CSV file
2. **Data Cleaning** 
   - ✅ Remove duplicates
   - ✅ Fix product names (standardize, remove extra spaces)
   - ✅ Drop returns/cancellations (negative quantities, cancelled invoices)
   - ✅ Filter rare items (products with low transaction frequency)
   - ✅ Remove empty baskets
3. **Exploratory Data Analysis (EDA)**
   - Most sold products
   - Basket size distribution
   - Top product pairs (co-occurrence)
   - Sales over time
4. **Transaction Encoding** - One-hot encoding for transactions
5. **Data Split** - Split data into train/test sets for validation
6. **Association Rule Mining**
   - Apriori algorithm (classic)
   - FP-Growth algorithm (faster, recommended)
   - Configurable parameters: min_support, min_confidence, min_lift
7. **Rule Generation & Filtering**
   - Filter by lift > 1 (positive association)
   - High confidence (reliable predictions)
   - Reasonable support (sufficient data)
8. **Visualization**
   - Network graphs of product associations
   - Heatmaps (support, confidence, lift)
   - Scatter plots (support vs confidence)
   - Item co-occurrence matrices
   - Rule metrics distribution

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

Key libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `mlxtend` - Association rule mining (Apriori, FP-Growth)
- `matplotlib` & `seaborn` - Visualization
- `networkx` - Network graph visualization
- `scikit-learn` - Data splitting

## Usage

1. **Prepare your data**: Ensure your CSV file has columns:
   - `InvoiceNo` - Transaction/invoice identifier
   - `Description` - Product name/description
   - `Quantity` - Product quantity (positive values)
   - `InvoiceDate` - Transaction date (optional, for time analysis)

2. **Update data path**: In the notebook, update the CSV file path:
   ```python
   df = pd.read_csv('your_data_file.csv')
   ```

3. **Run the notebook**: Execute cells sequentially to:
   - Load and clean data
   - Perform EDA
   - Generate association rules
   - Visualize results

4. **Adjusted parameters** :
   ```python
   min_support = 0.01      # Minimum support (1% of transactions)
   min_confidence = 0.3    # Minimum confidence (30%)
   min_lift = 1.0          # Minimum lift (positive association)
   ```

## Output

The analysis generates:
- **Association rules CSV** - All business-focused rules with metrics
- **Frequent itemsets CSV** - All frequent product combinations
- **Visualizations** - Network graphs, heatmaps, and distribution plots
- **Summary statistics** - Key insights and top rules

## Key Metrics Explained

- **Support**: Frequency of itemset appearing in transactions
- **Confidence**: Probability of consequent given antecedent
- **Lift**: Strength of association (lift > 1 = positive association)
- **Conviction**: Measure of dependency (higher = stronger rule)

## Applications

- **Product Recommendations**: Suggest complementary products
- **Store Layout Optimization**: Place related products near each other
- **Marketing Strategies**: Bundle products, cross-selling campaigns
- **Inventory Management**: Stock related products together
- **Pricing Strategies**: Price related products competitively

## Algorithm Comparison

- **Apriori**: Classic algorithm, good for small-medium datasets
- **FP-Growth**: Faster, more efficient for large datasets (recommended)
- **Eclat**: Alternative algorithm (can be implemented if needed)

## Notes

- The notebook automatically handles data cleaning and preprocessing
- Rare items are filtered to reduce noise (default: min 10 transactions)
- Rules are filtered for business relevance (lift > 1, confidence >= 0.5)
- Results are exported to CSV for further analysis


# Amazon Product Rating Prediction

This project predicts Amazon product ratings using a machine learning pipeline built in a Jupyter notebook. It combines product text (name, category, description, reviews) with numeric and categorical features to train a regression model, then evaluates and visualizes its performance.

## Project files

- `Amazon_Product_Rating_Prediction.ipynb` - end-to-end workflow (data cleaning, EDA, feature engineering, training, evaluation).
- `amazon.csv (1).zip` - dataset used by the notebook.

## Dataset

The notebook expects an Amazon products/reviews CSV with (at least) the following columns:

| Column | Description |
|---|---|
| `product_id` | Unique product identifier |
| `product_name` | Product title |
| `category` | Product category (often nested, e.g. `Electronics\|Mobiles\|...`) |
| `discounted_price` | Current/sale price (raw text, may include currency symbols) |
| `actual_price` | Original list price |
| `discount_percentage` | Discount applied, as text (e.g. `"40%"`) |
| `rating` | Target variable - product star rating |
| `rating_count` | Number of ratings received |
| `about_product` | Product description bullet points |
| `review_title` | Review headline text |
| `review_content` | Full review text |

Any missing expected column is auto-created as `NaN` so the notebook doesn't break on slightly different dataset versions.

## What the notebook does

1. **Load data** - reads the zipped CSV and prints its shape/preview.
2. **Clean data** - strips currency symbols/commas from price, count, and rating fields and converts them to numeric types; standardizes column names to lowercase; drops duplicate rows; drops rows with a missing target (`rating`).
3. **Feature engineering**:
   - `combined_text` - concatenation of product name, category, description, review title, and review content, used for TF-IDF vectorization.
   - `review_length` / `about_length` - word counts of review and description text.
   - `discount_ratio` - `discounted_price / actual_price`.
4. **Exploratory Data Analysis (EDA)** - generates plots for:
   - Rating distribution
   - Top 10 product categories
   - Discount percentage distribution
   - Actual and discounted price distributions
   - Discount percentage vs. rating scatter plot
   - Correlation heatmap across numeric features
5. **Preprocessing pipeline** (via `ColumnTransformer`):
   - Text → `TfidfVectorizer` (max 5,000 features, English stop words removed)
   - Categorical (`category`) → most-frequent imputation + one-hot encoding
   - Numeric (prices, discount, rating count, text lengths) → median imputation
6. **Model training** - fits a `Ridge` regression model (`alpha=1.0`) on an 80/20 train-test split.
7. **Evaluation** - reports MAE, RMSE, and R² on the test set.
8. **Result visualization**:
   - Actual vs. predicted rating scatter plot
   - Residual distribution plot
   - Table of sample predictions

## Requirements

Install Python packages used in the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to run

1. Ensure `amazon.csv (1).zip` is available in the same directory as the notebook.
2. Open and run `/home/runner/work/Amazon-Product-Rating-Prediction/Amazon-Product-Rating-Prediction/Amazon_Product_Rating_Prediction.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order.

## Output

Running the notebook produces:

- Console output: dataset shape, cleaned shape, `df.info()`, model performance metrics (MAE, RMSE, R²), and a table of 15 sample actual-vs-predicted ratings.
- Inline plots: rating distribution, top categories, price/discount distributions, correlation heatmap, actual vs. predicted scatter plot, and residual histogram.

## Project structure

```
.
├── Amazon_Product_Rating_Prediction.ipynb
├── amazon.csv (1).zip
└── README.md
```

## Possible improvements

- Replace `Ridge` with a tree-based model (e.g. `RandomForestRegressor`, `XGBRegressor`) to capture non-linear relationships.
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`) for the Ridge `alpha` or alternative models.
- Use cross-validation instead of a single train-test split for more robust performance estimates.
- Split the nested `category` field into separate hierarchy levels as additional categorical features.
- Experiment with sentence embeddings instead of TF-IDF for the review/description text.

## Notes

- The notebook currently reads the dataset from `"/content/amazon.csv (1).zip"`, which is a Colab-style path.
- If running locally, update the dataset path in the notebook to your local file location.

## License

Add a license of your choice (e.g. MIT) if you plan to share or open-source this project.

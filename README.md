# Amazon Product Rating Prediction

This project predicts Amazon product ratings using a machine learning pipeline built in a Jupyter notebook.

## Project files

- `Amazon_Product_Rating_Prediction.ipynb` - end-to-end workflow (data cleaning, EDA, feature engineering, training, evaluation).
- `amazon.csv (1).zip` - dataset used by the notebook.

## What the notebook does

1. Loads and cleans Amazon product data.
2. Standardizes columns and handles missing values.
3. Builds text, numeric, and categorical features.
4. Trains a Ridge regression model using a `scikit-learn` pipeline.
5. Evaluates performance with MAE, RMSE, and R².
6. Visualizes distributions, relationships, and residuals.

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

## Notes

- The notebook currently reads the dataset from `"/content/amazon.csv (1).zip"`, which is a Colab-style path.
- If running locally, update the dataset path in the notebook to your local file location.
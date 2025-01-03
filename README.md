# URL Phishing Detection

This repository contains a machine learning project for detecting phishing URLs based on their structural and behavioral features. Phishing URLs are malicious links designed to steal sensitive information from users, and this project aims to build an effective model to classify URLs as phishing or legitimate.

## Overview

The project uses a supervised machine learning approach, leveraging structured datasets, feature extraction techniques, and model optimization to classify URLs. The dataset was preprocessed, models were trained and evaluated, and hyperparameters were tuned to improve performance.

## Algorithm Used

The primary algorithm used in this project is **Random Forest Classifier**, a robust ensemble learning method that operates by constructing multiple decision trees and outputting the class with the majority vote. Random Forest is effective for classification tasks, particularly when the data is moderately imbalanced.

<h3>Why Random Forest?</h3>

- Handles both categorical and numerical features.

- Reduces overfitting by averaging multiple decision trees.

- Provides feature importance scores to understand which features contribute most to the predictions.

## Dataset

The dataset used for training and testing this project was obtained from [Kaggle](https://www.kaggle.com/datasets/simaanjali/phising-detection-dataset). It contains various features extracted from URLs, such as:

- Structural Features: URL length, number of dots, dashes, and numeric characters.

- Behavioral Indicators: Presence of IP addresses, @ symbols, and HTTPS in the hostname.

- Path Information: Depth of directory hierarchy and number of segments in the URL path.

- The target variable indicates whether a URL is phishing (1) or legitimate (0).

The target variable indicates whether a URL is phishing (1) or legitimate (0).

## Data Preprocessing

1. Splitting the Data:
   - The dataset was split into training (80%) and testing (20%) sets.
   - Stratified splitting was used to maintain the class distribution across train and test sets.
2. Addressing Imbalance:
   - The dataset was imbalanced, with more legitimate URLs (class 0) than phishing URLs (class 1).
   - Under-sampling was initially performed to balance the classes by randomly reducing the majority class (legitimate URLs).

## Model Training and Optimization

<h3>First Attempt: Initial Parameters</h3>

- The Random Forest model was trained using the following parameters:

  - `n_estimators`: 100
  - `max_depth`: 15
  - `min_samples_split`: 15
  - `max_leaf_nodes`: 100
  - `n_estimators`: 100
  - `class_weight`: `balanced`

<h3>Performance Result</h3>

- The initial model achieved a moderate performance:

  - **Precision (Phishing)**: 0.77
  - **Recall (Phishing)**: 0.84
  - **F1-Score (Phishing)**: 0.81

<h3>Threshold Adjustment</h3>

To improve the balance between precision and recall, a custom classification threshold was applied. This adjustment slightly improved the precision but had a minimal impact on overall performance:
- **Precision (Phishing)**: 0.74

- **Recall (Phishing)**: 0.90
  
- **F1-Score (Phishing)**: 0.81

While the recall improved, the precision is slightly deteriorated.

<h3>Hyperparameter Tuning with Grid Search</h3>

To overcome the limitations of threshold adjustment, a grid search was performed to find the best combination of hyperparameters for the Random Forest model. The search included:

- `n_estimators`: [100, 200, 500]
  
- `max_depth` : [10, 15, 20, None]
  
- `min_samples_split`: [2, 5, 10, 15]
  
- `min_samples_leaf`: [1, 5, 10, 15]

- `max_leaf_nodes`: [50, 100, None]

- `class_weight`: ['balanced', {0: 1, 1: 2}]

**Best Parameters Found:**

- `class_weight`: {0: 1, 1: 2}
  
- `max_depth`: None

- `min_samples_split`: 10

- `min_samples_leaf`: 1
  
- `max_leaf_nodes`: None

- `n_estimators`: 500

- `random_state`: 42

<h3>Final Model</h3>

After applying the best parameters, the model was retrained, and a custom threshold was applied to further balance precision and recall. This resulted in significant improvements:

- **Final Performance:**

  - **Precision (Phishing)**: 0.77
  - **Recall (Phishing)**: 0.93
  - **F1-Score (Phishing)**: 0.84

## Testing the Model Using Real URLs

The trained model can be tested on a real-world dataset of URLs formatted similarly to the train-test dataset. For this project, an additional dataset of URLs was obtained from [Kaggle](https://www.kaggle.com/datasets/ashharfarooqui/phising-urls). Each URL in the dataset was processed to extract the same features as those used during model training and testing. The features include:

- Structural features (e.g., URL length, number of dots and dashes).

- Behavioral indicators (e.g., presence of IP addresses and HTTPS in hostname).

- Path-based features (e.g., path depth and path length).

After feature extraction, the URLs were passed through the trained model to predict whether they were phishing (1) or legitimate (0). The predictions are stored in a new column, "Predicted," for analysis and validation.

## How to Use This Repository

Clone the repository by  running the below line of code in your terminal:
```
git clone https://github.com/bitacode/URL-Phishing-Detection.git
```

## Results and Insights

1. Model Insights:
   
   The Random Forest model effectively identifies phishing URLs, with reasonable trade-offs between precision and recall after threshold adjustment.

3. Challenges:
   
   Class imbalance initially hindered performance, but under-sampling and hyperparameter tuning improved results significantly.

5. Future Work:
   - Explore other models like Gradient Boosting.
   - Investigate advanced feature engineering or ensemble methods for further improvements.

# Tabular Data Regression Project

This project implements a regression model to predict a target variable based on 53 anonymized features.

## Setup

1. Clone this repository:

   ```
   git clone https://github.com/your-username/tabular-regression-project.git
   cd tabular-regression-project
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `data/`: Contains the training and test datasets
- `notebooks/`: Jupyter notebook with exploratory data analysis
- `src/`: Python scripts for training and prediction
- `requirements.txt`: List of required Python packages
- `README.md`: This file
- `predictions.csv`: File with prediction results (generated after running predict.py)
- `model_results.csv`: CSV file containing performance metrics for different models

## Usage

1. Exploratory Data Analysis:

   - Open and run the Jupyter notebook in the `notebooks/` directory.

2. Train the model:

   ```
   python src/train.py
   ```

3. Make predictions:
   ```
   python src/predict.py
   ```

The predictions will be saved in `predictions.csv` in the project root directory.

## Model Details

The current implementation trains and evaluates multiple regression models:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Random Forest Regressor

These models are trained on two different feature sets:

1. All available features
2. Only features 6 and 7

## Training Process

The training process involves the following steps:

1. Data loading and preprocessing
2. Splitting the data into training and validation sets
3. Feature scaling using StandardScaler
4. Training each model using cross-validation
5. Evaluating models on the validation set
6. Selecting the best-performing model based on validation RMSE

## Model Evaluation

The models are evaluated using Root Mean Square Error (RMSE) on both the training (via cross-validation) and validation sets. RMSE is calculated and logged during the training process, providing a measure of the average prediction error in the same units as the target variable.

## Results

The performance metrics for each model are saved in `model_results.csv`. Here's a summary of the results:

| Model             | Mean CV RMSE | Std CV RMSE | Validation RMSE | Description                 |
| ----------------- | ------------ | ----------- | --------------- | --------------------------- |
| Linear Regression | 28.8866      | 0.0253      | 29.0154         | Using all features          |
| Lasso             | 28.8709      | 0.0239      | 29.0038         | Using all features          |
| Ridge             | 28.8866      | 0.0253      | 29.0154         | Using all features          |
| Random Forest     | 0.0049       | 0.0001      | 0.0038          | Using all features          |
| Linear Regression | 28.8716      | 0.0233      | 29.0000         | Using only features 6 and 7 |
| Lasso             | 28.8709      | 0.0239      | 29.0038         | Using only features 6 and 7 |
| Ridge             | 28.8716      | 0.0233      | 29.0000         | Using only features 6 and 7 |
| Random Forest     | 0.0020       | 0.0000      | 0.0016          | Using only features 6 and 7 |

### Analysis of Results

1. Performance comparison:

   - The Random Forest model significantly outperforms all other models, achieving much lower RMSE values.
   - Linear models (Linear Regression, Lasso, and Ridge) perform similarly, with RMSE values around 29.

2. Feature set comparison:

   - For linear models, using only features 6 and 7 yields slightly better results than using all features.
   - For the Random Forest model, using only features 6 and 7 provides even better results than using all features.

3. Best performing model:

   - The Random Forest model using only features 6 and 7 achieves the lowest Validation RMSE of 0.0016, making it the best-performing model in this experiment.

4. Consistency:

   - The Random Forest model shows very low standard deviation in cross-validation RMSE, indicating consistent performance across different subsets of the data.

5. Overfitting consideration:
   - The extremely low RMSE values for the Random Forest model might indicate potential overfitting. Further investigation and possibly regularization techniques may be needed to ensure generalization.

Based on these results, the Random Forest model using only features 6 and 7 is selected as the best model for this regression task. However, it's important to monitor its performance on unseen data to ensure it generalizes well.

# MNIST Digit Classification Project

This project implements multiple classifiers for the MNIST handwritten digit dataset using different algorithms.

## Usage

Run the main script:

```
python src/mnist_classifier.py
```

## How the Code Works

1. The code defines an abstract base class `DigitClassificationInterface` that serves as an interface for all classifier models.

2. Three classifier models are implemented, each inheriting from `DigitClassificationInterface`:

   - `CNNModel`: A Convolutional Neural Network using TensorFlow/Keras
   - `RandomForestModel`: A Random Forest classifier using scikit-learn
   - `RandomModel`: A model that makes random predictions

3. The `DigitClassifier` class acts as a facade, allowing the user to choose which algorithm to use by specifying 'cnn', 'rf', or 'rand'.

4. The script loads and preprocesses the MNIST dataset, then trains and evaluates each classifier.

## Results

Here are the results from running the script:

```
CNN accuracy: 0.9891
RF accuracy: 0.9689
RAND accuracy: 0.0995
```

### Analysis

- The CNN model achieves the highest accuracy at 98.91%.
- The Random Forest model also performs well with 96.89% accuracy.
- As expected, the Random model performs poorly with about 10% accuracy (close to random guessing for 10 classes).
- Both CNN and RF correctly predict the first test image as 7.

These results demonstrate the effectiveness of both the CNN and Random Forest approaches for the MNIST digit classification task, with the CNN slightly outperforming the Random Forest model.

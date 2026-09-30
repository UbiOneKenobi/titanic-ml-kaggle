# Titanic Survival Prediction

A machine learning project based on the Kaggle Titanic dataset. The goal is to predict whether a passenger survived the Titanic disaster using passenger information such as age, sex, passenger class, fare, family size, cabin location, and ticket information.

This project was built as a practical exercise in applying an end-to-end supervised machine learning workflow with `pandas` and `scikit-learn`.

## Project Workflow

The project follows these main steps:

1. Exploratory data analysis
2. Feature engineering
3. Data preprocessing
4. Model comparison using cross-validation
5. Hyperparameter tuning
6. Final model training
7. Kaggle submission

## Feature Engineering

Several new features were created from the original dataset:

- **FamSize**: total number of family members travelling together, calculated from `SibSp` and `Parch`.
- **Title**: passenger title extracted from the `Name` column.
- **Corridor**: first character of the `Cabin` value, used as an approximation of the passenger's deck/area.
- **TicketGroupSize**: number of passengers sharing the same ticket, calculated with `groupby()` and `transform("count")`.

After creating these features, the original columns that were no longer needed (`Name`, `Cabin`, `Ticket`, `SibSp`, and `Parch`) were removed from the model input.

## Preprocessing

The preprocessing pipeline is built with `ColumnTransformer` and separate transformations for different feature types.

### Categorical features

- `Sex`
- `Embarked`
- `Title`
- `Corridor`

Categorical variables are encoded with `OneHotEncoder`.

For `Corridor`, missing values are replaced with an `"Unknown"` category before one-hot encoding.

`handle_unknown="ignore"` is used so that unseen categories in validation or test data do not cause errors.

### Numerical features

- `Pclass`
- `Age`
- `Fare`
- `FamSize`
- `TicketGroupSize`

Numerical features are standardized with `StandardScaler`.

Missing values in `Age` and `Fare` are filled using the median before scaling.

## Models Tested

Several classification models were evaluated using cross-validation:

- K-Nearest Neighbors
- SGD Classifier
- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression

The strongest validation performance was obtained with **Logistic Regression**, with cross-validation accuracy of approximately **82–83%**.

## Hyperparameter Tuning

Two tuning approaches were explored:

### K-Nearest Neighbors

`GridSearchCV` was used to test different values for:

- `n_neighbors`
- `weights`

The tuning did not significantly improve the original KNN performance.

### Logistic Regression

`RandomizedSearchCV` was used to tune the regularization parameter `C` using a log-uniform distribution.

The final Logistic Regression model used:

- `solver="liblinear"`
- `penalty="l2"`
- optimized `C`

## Final Model

The final model is a complete scikit-learn pipeline containing:

```text
Feature preprocessing
        ->
Logistic Regression
```

Keeping preprocessing and the classifier in the same pipeline ensures that transformations are learned only from the training data during cross-validation and are applied consistently to new data.

## Kaggle Submission

The trained model is used to generate predictions for the Kaggle test set and create a submission file with the required structure:

```text
PassengerId,Survived
892,0
893,1
...
```

The project achieved a Kaggle score of approximately **0.756** on the submitted test predictions.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Google Colab
- Kaggle

## Repository Structure

```text
.
├── titanic.py
├── README.md
└── submission.csv   # optional / generated output
```

The Kaggle `train.csv` and `test.csv` files are not included in the repository and should be downloaded separately from the Titanic competition page.

## Running the Project

Install the required Python packages:

```bash
pip install pandas numpy scipy scikit-learn
```

Update the dataset paths if necessary:

```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

Then run:

```bash
python titanic.py
```

## What I Learned

This project was primarily built to practice the machine learning workflow rather than to maximize leaderboard performance. It helped reinforce concepts such as:

- exploratory data analysis for classification problems
- feature engineering from raw categorical data
- missing-value imputation
- one-hot encoding
- feature scaling
- `ColumnTransformer` and scikit-learn pipelines
- cross-validation
- model comparison
- Grid Search and Randomized Search
- avoiding preprocessing leakage during validation
- generating predictions for unseen data

## Future Improvements

Possible next steps include:

- grouping rare passenger titles into a single category
- testing alternative representations of family size
- improving ticket-based feature engineering
- comparing additional ensemble models
- using nested cross-validation for a more rigorous estimate after hyperparameter tuning
- analyzing precision, recall, F1-score, and the confusion matrix in addition to accuracy

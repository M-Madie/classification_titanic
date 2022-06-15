import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd

# load data, delete unwanted Columns
df = pd.read_csv('week2/train.csv', index_col=0)
del df['Ticket']
del df['Cabin']
del df['Name']

# Imputation - Column[Embarked]
imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = pd.DataFrame(imputer.fit_transform(df[['Embarked']]))

# split to X, and y
X = df.iloc[:,1:]
y = df['Survived']

# feature engineering for numericals
numeric_feature_age = ["Age"]
numeric_feature_Fare = ["Fare"]
# Age
age_numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), 
    KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    )
# Fare
fare_numeric_transformer = make_pipeline(
    MinMaxScaler()
    )


# feature engineering for categorical
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("age", age_numeric_transformer, numeric_feature_age),
        ("fare", fare_numeric_transformer, numeric_feature_Fare),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder='passthrough')

# create the model pipeline
pipeline = make_pipeline(preprocessor, LogisticRegression(max_iter=300))

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 20)

####
# fit the pipeline to training data
pipeline.fit(X_train, y_train)
print("model score(Train): %.3f" % pipeline.score(X_train, y_train))

###
# calculate the accuracy score from test data
print("model score(Test-Validation): %.3f" % pipeline.score(X_test, y_test))

# get predictions from the pipeline
print(pipeline.predict(X_test))

# get prediction probabilities from the pipeline 
print(pipeline.predict_proba(X_test)[:, 1])


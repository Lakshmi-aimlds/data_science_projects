import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer

input_data_path = os.path.join("/opt/ml/processing/input", "processed_data.csv")
data = pd.read_csv(input_data_path)

target = 'Sales'
numeric_features = ['TV','Radio','Newspaper']

X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)
pd.DataFrame(X_test).to_csv(test_features_output_path, header=False, index=False)

y_train.to_csv(train_labels_output_path, header=False, index=False)
y_test.to_csv(test_labels_output_path, header=False, index=False)
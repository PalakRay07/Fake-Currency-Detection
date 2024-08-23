

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load and format data
data = pd.read_csv('C:/Users/super/Downloads/data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']

# Print data info
print(data.head())
print(data.info)

# Pairplot for feature exploration
sns.pairplot(data, hue='auth')
plt.show()

# Distribution of target variable
plt.figure(figsize=(8, 6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'])

# Annotate target counts (corrected line)
target_count = data.auth.value_counts()
plt.annotate(text=str(target_count[0]), xy=(-0.04, 10 + target_count[0]), size=14)  # Added 'text' argument
plt.annotate(text=str(target_count[1]), xy=(0.96, 10 + target_count[1]), size=14)  # Added 'text' argument
plt.ylim(0, 900)
plt.show()

# Undersample majority class
nb_to_delete = target_count[0] - target_count[1]
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
data = data[nb_to_delete:]

print(data['auth'].value_counts())

# Separate features and target
x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize features
scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

# Train logistic regression model
clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

# Make predictions
y_pred = np.array(clf.predict(x_test))

# Confusion matrix
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                          columns=["Pred.Negative", "Pred.Positive"],
                          index=['Act.Negative', "Act.Positive"])

# Calculate and print accuracy
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = round((tn + tp) / (tn + fp + fn + tp), 4)
print(conf_mat)
print(f'\n Accuracy = {round(100 * accuracy, 2)}%')

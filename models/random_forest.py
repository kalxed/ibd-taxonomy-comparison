from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ibd_data = pd.read_csv("./data/hmp/normalized_hmp_final.csv", index_col=0)
ibd_data["Label"] = 1
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
hmp_data = pd.read_csv("./data/ibdmdb/normalized_ibd_final.csv", index_col=0)
hmp_data["Label"] = 0
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

combined_data = pd.concat([ibd_data, hmp_data], axis=0).dropna()

X = combined_data.drop("Label", axis=1)
y = combined_data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1)

model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=10)
print(scores)
print("Mean: ",scores.mean())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model

print("Score: ", model.score(X_test, y_test))


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "IBD"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Grab the false positive
fp = cm[0][1]
# Plot the false positive sample from X
fp_sample = X_test[y_test == 0].iloc[fp]
plt.figure(figsize=(12, 6))
plt.bar(X_test.columns, fp_sample)
plt.title("False Positive Sample")
plt.show()

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances
important_features = feature_importances[feature_importances['Importance'] > 0.01]
plt.figure(figsize=(10, 6))
sns.barplot(data=important_features, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importances (Above 0.01)")
plt.show()
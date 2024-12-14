from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
ibd_data = pd.read_csv("./data/hmp/normalized_hmp_data.csv", index_col=0)
ibd_data["Label"] = 1
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
hmp_data = pd.read_csv("./data/ibdmdb/normalized_ibd_data.csv", index_col=0)
hmp_data["Label"] = 0
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

combined_data = pd.concat([ibd_data, hmp_data], axis=0).dropna()

X = combined_data.drop("Label", axis=1)
y = combined_data["Label"]

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1)

# Create the model

model = SVC(kernel='linear', C=1.0)
scores = cross_val_score(model, X, y, cv=10)
print(scores)
print("Mean: ",scores.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Score: ", model.score(X_test, y_test))

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "IBD"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Analyze Important Features
if model.kernel == 'linear':
    feature_importance = abs(model.coef_).mean(axis=0)

    # Filter features with importance > 0.5
    important_features = [(name, importance) for name, importance in zip(feature_names, feature_importance) if importance > 0.5]

    if important_features:
        # Unpack feature names and values
        important_feature_names, important_values = zip(*important_features)

        # Plot filtered feature importances
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(important_values)), important_values)
        plt.xticks(range(len(important_feature_names)), important_feature_names, rotation=45, ha='right')
        plt.title("Important Features (SVC Coefficients > 0.5)")
        plt.ylabel("Average Magnitude")
        plt.show()
    else:
        print("No features have an importance magnitude > 0.5")

# Support Vectors Analysis
print(f"Number of Support Vectors per Class: {model.n_support_}")
print("Support Vectors Indices:", model.support_)

# Permutation Importance (for any SVC kernel)
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
perm_importances = result.importances_mean

plt.figure(figsize=(12, 6))
plt.bar(range(len(perm_importances)), perm_importances)
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.title("Permutation Importance (Test Set)")
plt.ylabel("Importance")
plt.show()
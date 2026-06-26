import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot an annotated confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    group_names = ['True Normal', 'False Suspect', 'False Pathological',
                   'False Normal', 'True Suspect', 'False Pathological',
                   'False Normal', 'False Suspect', 'True Pathological',]
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_pcts   = [f"{v:.2%}"  for v in cm.flatten() / np.sum(cm)]
    labels = np.array(
        [f"{n}\n{c}\n{p}" for n, c, p in zip(group_names, group_counts, group_pcts)]
    ).reshape(3, 3)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title, top_n=10):
    """Bar chart of the top-N feature importances for a fitted RF model."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    top.plot(kind='barh', color=sns.color_palette("viridis", top_n))
    plt.xlabel("Gini Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

data = pd.read_csv('fetal_health.csv', index_col=0)
print(data.describe())
print(data.info())

print('\nUnique data for severe_decelerations:', np.unique(data['severe_decelerations']))
print('\nUnique data for histogram_tendency:', np.unique(data['histogram_tendency']))

#Convert features into categorical
data['severe_decelerations'] = data['severe_decelerations'].astype('category')
data['histogram_tendency'] = data['histogram_tendency'].astype('category')
health_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
data['fetal_health'] = data['fetal_health'].map(health_mapping).astype('category')

print(data['fetal_health'].value_counts(normalize=True))

X = data.drop(columns=['fetal_health']).copy()
Y = data['fetal_health'].cat.codes

X['severe_decelerations'] = X['severe_decelerations'].cat.codes
X['histogram_tendency'] = X['histogram_tendency'].cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#### DECISION TREE ####
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=4)
decision_tree.fit(X_train, y_train)
pred_tree = decision_tree.predict(X_test)

plot_confusion_matrix(y_test, pred_tree, title='Decision Tree Confusion Matrix')
print('Decision Tree Classification Report:\n',
      classification_report(y_test, pred_tree, target_names=['Normal', 'Suspect', 'Pathological']))

plot_feature_importance(
    decision_tree,
    X_train.columns,
    "Feature Importances – Model 1 (Decision Tree)",
)

#Obtain an accuracy of 0.91

#### RANDOM FOREST ####
param_grid = {
    'max_features': [2,4,6,8,10,12,14,16],
    'n_estimators': [50,100,150,200,250,300]
}
rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    n_jobs=-1,
    cv=5,
    scoring='accuracy'
)

rf.fit(X_train, y_train)

print(f"Best Parameters: {rf.best_params_}")
print(f"Best CV Accuracy: {rf.best_score_:.4f}")

rf = rf.best_estimator_

pred_rf = rf.predict(X_test)
plot_confusion_matrix(y_test, pred_rf, title='Random Forest Confusion Matrix')
print('Random Forest Classification Report:\n',
      classification_report(y_test, pred_rf, target_names=['Normal', 'Suspect', 'Pathological']))
plot_feature_importance(
    rf,
    X_train.columns,
    'Feature Importances - Model 2 (Random Forest)'
)

#Obtain an accuracy of 0.94

#### LIGHT BOOST ####
gbm_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'min_samples_leaf': [5, 10, 15],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1, 2]
}

gbm = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=123),
    param_grid=gbm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

gbm.fit(X_train, y_train)

print(f"Best GBM Parameters: {gbm.best_params_}")

# Predict and Evaluate
gbm = gbm.best_estimator_
pred_gbm = gbm.predict(X_test)
plot_confusion_matrix(y_test, pred_gbm, title='Gradient Boosting Confusion Matrix')
print('Gradient Boosting Classification Report:\n',
      classification_report(y_test, pred_rf, target_names=['Normal', 'Suspect', 'Pathological']))
plot_feature_importance(
    gbm,
    X_train.columns,
    'Feature Importances - Model 3 (Gradient Boosting)'
)

#Accuracy of 0.94 but increase the recall for 'Pathological' to 0.73
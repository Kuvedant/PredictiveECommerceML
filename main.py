import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.decomposition import PCA
from tqdm import tqdm  # Import tqdm for the progress bar

from clean_preprocess import load_data, preprocess_data, split_and_scale_data
from EDA import plot_revenue_distribution, plot_correlation_matrix, plot_numerical_stats
from models import classifiers, param_grid

# Ensure results and models directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

def main():
    print("Step 1: Loading Data")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    data = load_data(url)
    print("Data Loaded")

    print("Step 2: Preprocessing Data")
    preprocessed_data = preprocess_data(data)
    print("Data Preprocessed")

    print("Step 3: Performing Exploratory Data Analysis (EDA)")
    plot_revenue_distribution(data)
    plot_correlation_matrix(preprocessed_data)
    plot_numerical_stats(preprocessed_data)
    print("EDA Completed")

    print("Step 4: Splitting and Scaling Data")
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(preprocessed_data)
    print("Data Split and Scaled")

    print("Step 5: Applying PCA")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_train_extended = np.concatenate([X_train_scaled, X_train_pca], axis=1)
    X_test_extended = np.concatenate([X_test_scaled, X_test_pca], axis=1)
    print("PCA Applied")

    print("Step 6: Training and Evaluating Models")
    scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score)}
    results = {}
    best_model = None
    best_accuracy = 0

    for name, clf in tqdm(classifiers.items(), desc="Training Models"):
        print(f"Training {name} model...")
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid[name], scoring=scoring, refit='accuracy', return_train_score=False, cv=5)
        grid_search.fit(X_train_extended, y_train)
        scores = cross_val_score(grid_search.best_estimator_, X_test_extended, y_test, cv=5, scoring='accuracy')
        results[name] = scores

        if scores.mean() > best_accuracy:
            best_accuracy = scores.mean()
            best_model = grid_search.best_estimator_

        model_path = os.path.join('models', f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        print(f"{name} model trained and saved.")

    print("Model Training and Evaluation Completed")

    print("Step 7: Visualizing Classifier Performance")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=pd.DataFrame(results), orient='h', palette='Set2')
    ax.set_title('Comparison of Classifier Performance')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Classifiers')
    plt.savefig('results/classifier_performance.png')
    plt.close()
    print("Classifier Performance Visualized")

    print("Step 8: Generating Confusion Matrix and AUC-ROC Curve for Best Model")
    best_model.fit(X_train_extended, y_train)
    y_pred = best_model.predict(X_test_extended)
    y_prob = best_model.predict_proba(X_test_extended)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve for {best_model.__class__.__name__}')
    plt.legend(loc='lower right')
    plt.savefig('results/auc_roc_curve.png')
    plt.close()
    print("AUC-ROC Curve Generated")

    print("Step 9: Calculating Feature Importance for Best Model")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = None

    if importances is not None:
        feature_names = list(preprocessed_data.columns) + ['PCA1', 'PCA2']
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {best_model.__class__.__name__}')
        plt.savefig('results/feature_importance.png')
        plt.close()
        print("Feature Importance Calculated and Visualized")

    print("All Steps Completed")

if __name__ == "__main__":
    main()

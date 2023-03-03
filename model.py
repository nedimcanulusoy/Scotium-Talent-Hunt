import os
import warnings
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from external_helpers import tuning_comparison

matplotlib.use('agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def model_(df):
    # Model to predict potential label with minimum error
    X = df.drop(["PLAYER_ID", "POTENTIAL_LABEL"], axis=1)
    y = df["POTENTIAL_LABEL"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    classifiers = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=5),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=5)
    }

    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(key)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("F1 Score: ", f1_score(y_test, y_pred))
        print("Precision: ", precision_score(y_test, y_pred))
        print("Recall: ", recall_score(y_test, y_pred))
        print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print("Classification Report: ", classification_report(y_test, y_pred))

        # Save accuracy score, f1 score, precision, recall, roc auc score, confusion matrix and classification report to txt file
        if not os.path.exists('best_models'):
            os.makedirs('best_models')

        with open(f"best_models/{key.replace(' ', '_')}_pre_tuning.txt", "w") as f:
            f.write("BEFORE HYPERPARAMETER TUNING\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
            f.write(f"F1 Score: {f1_score(y_test, y_pred)}\n")
            f.write(f"Precision: {precision_score(y_test, y_pred)}\n")
            f.write(f"Recall: {recall_score(y_test, y_pred)}\n")
            f.write(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}\n")
            f.write(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}\n")
            f.write(f"Classification Report: {classification_report(y_test, y_pred)}\n")
            f.write("#" * 75 + "\n\n")

        # Feature Importance for Random Forest Classifier and Logistic Regression
        if key == "Random Forest Classifier":
            feature_importance = pd.DataFrame(zip(X_train.columns, classifier.feature_importances_),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)

        elif key == "Logistic Regression":
            feature_importance = pd.DataFrame(zip(X_train.columns, classifier.coef_[0]),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)
        elif key == "XGBoost":
            feature_importance = pd.DataFrame(zip(X_train.columns, classifier.feature_importances_),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)


        # Create folder for feature importance under project
        if not os.path.exists('feature_importance'):
            os.makedirs('feature_importance')
            os.makedirs('feature_importance/confusion_matrix')
            os.makedirs('feature_importance/cross_validation')

        # Create a horizontal bar chart
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(key + ' Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'feature_importance/cross_validation/(PRE_CV)_{key.replace(" ", "_")}.png')

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'feature_importance/confusion_matrix/(PRE_CM)_{key.replace(" ", "_")}.png')

        print("-" * 50)

    # Grid search and Cross Validation
    param_grid = {
        "Random Forest Classifier": {
            "n_estimators": [50, 200, 500],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 8]
        },
        "Logistic Regression": {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "max_iter": [1000, 2000, 3000]
        },
        "XGBoost": {
            "n_estimators": [50, 200, 500],
            "max_depth": [5, 10, None],
            "learning_rate": [0.1, 0.01, 0.001]
        }
    }

    for key_, classifier_ in classifiers.items():
        print(f"\033[31mRunning grid search for {key_}...\033[0m")
        with tqdm(total=len(param_grid[key_]), desc="Grid Search Progress") as pbar:
            grid_search = GridSearchCV(estimator=classifier_, param_grid=param_grid[key_], cv=5, n_jobs=-1,
                                       return_train_score=True)
            for i, _ in enumerate(grid_search.fit(X_train, y_train).cv_results_['params']):
                pbar.update(1)  # update the progress bar for each parameter combination
            print(key_)
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Score: ", grid_search.best_score_)
        print("-" * 50)

        # Update the model with the best parameters
        classifier_.set_params(**grid_search.best_params_)
        classifier_.fit(X_train, y_train)

        # Accuracy, F1 Score, Precision, Recall, Confusion Matrix, Classification Report
        y_pred = classifier_.predict(X_test)
        print(key_)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("F1 Score: ", f1_score(y_test, y_pred))
        print("Precision: ", precision_score(y_test, y_pred))
        print("Recall: ", recall_score(y_test, y_pred))
        print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print("Classification Report: ", classification_report(y_test, y_pred))

        # Save accuracy score, f1 score, precision, recall, roc auc score, confusion matrix and classification report to txt file
        with open(f"best_models/{key_.replace(' ', '_')}_post_tuning.txt", "w") as f:
            f.write("AFTER HYPERPARAMETER TUNING\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
            f.write(f"F1 Score: {f1_score(y_test, y_pred)}\n")
            f.write(f"Precision: {precision_score(y_test, y_pred)}\n")
            f.write(f"Recall: {recall_score(y_test, y_pred)}\n")
            f.write(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}\n")
            f.write(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}\n")
            f.write(f"Classification Report: {classification_report(y_test, y_pred)}\n")

        # Save the model with the best parameters
        joblib.dump(classifier_, f"best_models/{key_.replace(' ', '_')}.pkl")

        # Feature Importance for Random Forest Classifier and Logistic Regression
        if key_ == "Random Forest Classifier":
            feature_importance = pd.DataFrame(zip(X_train.columns, grid_search.best_estimator_.feature_importances_),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)

        elif key_ == "Logistic Regression":
            feature_importance = pd.DataFrame(zip(X_train.columns, grid_search.best_estimator_.coef_[0]),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)

        elif key_ == "XGBoost":
            feature_importance = pd.DataFrame(zip(X_train.columns, grid_search.best_estimator_.feature_importances_),
                                              columns=["feature", "importance"]).sort_values("importance",
                                                                                             ascending=False)
            print("Feature Importance: ", feature_importance)
            print("#" * 50)

        # Create a horizontal bar chart
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(key_ + ' Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'feature_importance/cross_validation/(POST_CV)_{key_.replace(" ", "_")}.png')

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'feature_importance/confusion_matrix/(POST_CM)_{key_.replace(" ", "_")}.png')

        print("-" * 50)

    tuning_comparison()

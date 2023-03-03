import os


def dataframe_summary(dataframe, cat_threshold=1, card_threshold=3):
    # Categorical Variables
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_threshold and dataframe[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > card_threshold and str(dataframe[col].dtypes) in ["object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical Variables
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int', 'float']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}\n",
          f"Variables: {dataframe.shape[1]}\n",
          f'cat_cols: {len(cat_cols)}\n{cat_cols}\n',
          f'num_cols: {len(num_cols)}\n{num_cols}\n',
          f'cat_but_car: {len(cat_but_car)}\n{cat_but_car}\n',
          f'num_but_cat: {len(num_but_cat)}\n{num_but_cat}')

    return cat_cols, num_cols, cat_but_car


def tuning_comparison():
    # Append Logistic_Regression_post_tuning.txt to Logistic_Regression_pre_tuning.txt
    with open("best_models/Logistic_Regression_pre_tuning.txt", "a") as f:
        with open("best_models/Logistic_Regression_post_tuning.txt", "r") as f1:
            for line in f1:
                f.write(line)
            # Rename the file to Logistic_Regression_tuning.txt
            os.rename("best_models/Logistic_Regression_pre_tuning.txt", "best_models/Logistic_Regression_tuning.txt")
            # Delete the file Logistic_Regression_post_tuning.txt
            os.remove("best_models/Logistic_Regression_post_tuning.txt")

    # Append Random_Forest_Classifier_post_tuning.txt to Random_Forest_Classifier_pre_tuning.txt
    with open("best_models/Random_Forest_Classifier_pre_tuning.txt", "a") as f:
        with open("best_models/Random_Forest_Classifier_post_tuning.txt", "r") as f1:
            for line in f1:
                f.write(line)
            # Rename the file to Random_Forest_Classifier_tuning.txt
            os.rename("best_models/Random_Forest_Classifier_pre_tuning.txt",
                      "best_models/Random_Forest_Classifier_tuning.txt")
            # Delete the post tuning file
            os.remove("best_models/Random_Forest_Classifier_post_tuning.txt")

    # Append XGBoost_post_tuning.txt to XGBoost_pre_tuning.txt
    with open("best_models/XGBoost_pre_tuning.txt", "a") as f:
        with open("best_models/XGBoost_post_tuning.txt", "r") as f1:
            for line in f1:
                f.write(line)
            # Rename the file to XGBoost_Classifier_tuning.txt
            os.rename("best_models/XGBoost_pre_tuning.txt", "best_models/XGBoost_Classifier_tuning.txt")
            # Delete the post tuning file
            os.remove("best_models/XGBoost_post_tuning.txt")
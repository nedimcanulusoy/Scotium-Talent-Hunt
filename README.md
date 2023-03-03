# Scotium Talent Hunt with Machine Learning

---

This project aims to predict whether a football player is an average, below average or highlighted player based on their
characteristics and scores given by scouts. The dataset used in this project is from Scoutium, which includes the
features and scores of football players evaluated by the scouts.

---

Table of Contents

1. [Dataset](#dataset)
2. [Approach](#approach)
3. [Evaluation](#evaluation)
4. [Result](#result)
5. [How to Run](#how-to-run)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

### Dataset

The dataset used in this project is obtained from Scoutium, which includes information about football players'
characteristics observed in matches, evaluated by scouts. The datasets contain the following features:

1) scoutium_attributes:

- 8 attributes, 10.730 observations

    - **task_response_id**: The set of a scout's assessments of all players on a team's roster in a match
    - **match_id**: The match id
    - **evaulator_id**: The scout id
    - **player_id**: The player id
    - **position_id**: The position id
    - **analysis_id**: The set of a scout that contains attribute evaluations of a player in a match
    - **attribute_id**: The id of each attribute the players were evaluated on
    - **attribute_value**: Value (points) given by a scout to a player's attribute

2) scoutium_potential_labels:

- 5 attributes, 322 observations

    - **task_response_id**: The set of a scout's assessments of all players on a team's roster in a match
    - **match_id**: The match id
    - **evaulator_id**: The scout id
    - **player_id**: The player id
    - **potential_label**: The final label given by a scout to a player (target variable)

---

### How to Run

1) Clone the repository

```bash
git clone https://github.com/nedimcanulusoy/Scotium-Talent-Hunt-with-ML.git
```

2) Run the create_venv.py file to create a virtual environment

```bash
python create_venv.py
```

3) Activate the virtual environment

```bash
source venv/bin/activate
```

4) Install the required packages

```bash
pip install -r requirements.txt
```

5) Run the `scoutium_eda.ipynb` file for the Exploratory Data Analysis (EDA)

```bash
jupyter notebook scoutium_eda.ipynb
```

6) Run the `main.py` file for the model

```bash
python main.py
```

---

### Approach

The dataset is split into two sets, a training set and a test set, with 75% and 25% of the data, respectively.

The training set is used to train the model, and the test set is used to evaluate the model. The model is trained
using the following algorithms:

1) Random Forest
2) Logistic Regression
3) XGBoost

---

### Evaluation

The model is evaluated using the following metrics:

1) Accuracy
2) Precision
3) Recall
4) F1 Score
5) ROC AUC Score
6) Confusion Matrix
7) Classification Report
8) Feature Importance

**Note:** The evaluation results can be found in the `best_models` folder for each algorithm.

---

### Result

The best model is the Random Forest model with the following results:

| Metric        | Accuray | F1 Score | Precision | Recall | ROC AUC Score |
|---------------|---------|----------|-----------|--------|---------------|
| Before Tuning | 0.91 | 0.4  | 1.0 | 0.25 | 0.625 |
| After Tuning  | 0.92 | 0.61 | 0.8 | 0.5  | 0.74  |

The best parameters for the Random Forest model are:

| Parameter | Value |
|-----------|-------|
| n_estimators | 50    |
| max_depth | 10    |
| min_samples_split | 2     |

**Note:** The other models' information are also saved in the `best_models` folder.

---

### Contributing

1) Fork it (https://github.com/nedimcanulusoy/Scotium-Talent-Hunt-with-ML.git)
2) Create your feature branch (git checkout -b feature/your-feature)
3) Commit your changes (git commit -am 'Add some feature')
4) Push to the branch (git push origin feature/your-feature)
5) Create a new Pull Request

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

### Acknowledgements

These datasets are provided by [Scoutium](https://scoutium.com/), which is a football scouting platform that provides
football clubs, for this project. For this reason, they are not publicly available.




from sklearn.pipeline import Pipeline
from pre_process import pre_processed_data
from model import model_


def run_pipeline():
    pipeline = Pipeline([
        ('pre_process', pre_processed_data()),
        ('model', model_(pre_processed_data()))
    ])


if __name__ == "__main__":
    run_pipeline()

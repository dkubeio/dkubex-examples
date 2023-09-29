# Import necessary libraries
from flytekit import task, workflow, dynamic, Resources
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna
import mlflow
from typing import Tuple, Dict, Any
# Define a Flyte task for model training
@task(requests=Resources(cpu="2",mem="1Gi"))
def model_accuracy(n_estimators: int, max_depth: int, min_samples_split: float) -> float:
    import mlflow
    import mlflow.sklearn
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Initialize and train a Random Forest Classifier with given hyperparameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    # Perform cross-validation and return the mean accuracy
    accuracy = float(np.mean(cross_val_score(clf, X, y, cv=3)))
    return accuracy

# Define a Flyte task for the hyperparameter optimization
@task(requests=Resources(cpu="2",mem="1Gi"))
def optimize_hyp() -> Tuple[float, Dict[str, Any]]:
    import optuna
    import mlflow
    best_accuracy = 0.0
    best_params = {}
    # Define the objective function
    def objective(trial):
        nonlocal best_accuracy, best_params
        # Define the search space for hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
        
        run_name = f"OptunaTrial_{trial.number}"
        with mlflow.start_run(run_name=run_name):
            # Run the Flyte task with the selected hyperparameters
            accuracy = model_accuracy(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
            # Log any metrics or parameters as needed
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }
        return -accuracy  # Optuna minimizes, so negate the metric for maximization

    study = optuna.create_study(direction="maximize")  # For accuracy maximization
    study.optimize(objective, n_trials=20)  # You can specify the number of trials
    
    return best_accuracy, best_params 

@workflow
def optimize_model():
    best_accuracy = optimize_hyp()
    
    return best_accuracy

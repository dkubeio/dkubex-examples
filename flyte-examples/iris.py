# Import necessary libraries
from flytekit import task, workflow, dynamic, Resources
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define a Flyte task for model training
@task(requests=Resources(cpu="2",mem="1Gi"))
def train_model(n_estimators: int, max_depth: int, min_samples_split: float) -> float:
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Initialize and train a Random Forest Classifier with given hyperparameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    # Perform cross-validation and return the mean accuracy
    accuracy = float(np.mean(cross_val_score(clf, X, y, cv=3)))
    
    return accuracy

# Define a Flyte task for the hyperparameter optimization
@task(requests=Resources(cpu="2",mem="1Gi"))
def optimize_hyp() -> float:
    import optuna
    # Define the objective function
    def objective(trial):
        # Define the search space for hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)

        # Run the Flyte task with the selected hyperparameters
        accuracy = train_model(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

        return -accuracy  # Optuna minimizes, so negate the metric for maximization

    study = optuna.create_study(direction="maximize")  # For accuracy maximization
    study.optimize(objective, n_trials=100)  # You can specify the number of trials

    best_accuracy = -study.best_value  # Revert the negated value

    return best_accuracy

@workflow
def optimize_model():
    best_accuracy = optimize_hyp()

    best_params = {"n_estimators": 150, "max_depth": 32, "min_samples_split": 0.7}  # Set to 0 as it's not used in this example
    final_model_accuracy = train_model(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"])
    print(best_params)
    print(best_accuracy)
    print(final_model_accuracy)
    return best_params, best_accuracy, final_model_accuracy

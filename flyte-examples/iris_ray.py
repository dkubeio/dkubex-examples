from flytekit import task, workflow, Resources, dynamic
import mlflow.pyfunc
import time
from typing import List, Tuple, Dict, Any
import os 

# Set the MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = "http://d3x-controller.d3x.svc.cluster.local:5000"

experiment_name = os.environ.get("MLFLOW_EXP_NAME")
print(experiment_name)
# Define a Flyte task for model training
@task(requests=Resources(cpu="2",mem="1Gi"))
def model_accuracy(n_estimators: int, max_depth: int, min_samples_split: float) -> float:
    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

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
    import os
    experiment_name = "flyte_optuna"
    # Define the objective function
    def objective(trial):
        nonlocal best_accuracy, best_params, experiment_name
         
        # Define the search space for hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
        run_name = f"OptunaTrial_{trial.number}"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name, nested=True):
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
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="HPO_RUN") as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log the parent run ID as a parameter
        mlflow.log_param("parent_run_id", parent_run_id)
        study = optuna.create_study(direction="maximize")  # For accuracy maximization
        study.optimize(objective, n_trials=20)  # You can specify the number of trials
    
    return best_accuracy, best_params

@task(requests=Resources(cpu="2", mem="1Gi"))
def train_best_model(best_params: Dict[str, Any]) -> str:
    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import os
    experiment_name = "flyte_optuna"

    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Initialize and train a Random Forest Classifier with the best hyperparameters
    clf = RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"], random_state=42)
    accuracy = model_accuracy(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"]) 
    
    # Fit the model
    clf.fit(X, y)
    mlflow.set_experiment(experiment_name) 
    # Log the model to MLflow
    with mlflow.start_run(run_name="BestRandomForestModel") as run:
        mlflow.sklearn.log_model(clf, "best_random_forest_model")
        mlflow.log_metric("accuracy", accuracy)

        run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/best_random_forest_model"
    return model_uri

# Define a Flyte task for inference using Ray
@task(requests=Resources(cpu="2", mem="1Gi"))
def ray_inference(model_uri: str, data: List[List[float]]) -> Tuple[List[int], str]:
    import mlflow
    import mlflow.pyfunc
    import ray
    import time

    # Initialize Ray if it's not already
    if not ray.is_initialized():
        ray.init()
    # Load the best-trained model from MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    
    # Perform inference using Ray
    @ray.remote
    def inference(model, data):
        prediction = model.predict(data)
        return prediction.tolist()  # Pass data as a list
    
    results = ray.get(inference.remote(model, data))
    if results[0] == 0 :
        iris_type = "Setosa"
    elif results[0] == 1:
        iris_type = "Versicolour"
    else:
        iris_type = "Virginica"
    return results, iris_type


@workflow
def optimize_model() -> Tuple[float, List[int], str]:
    best_accuracy, best_params = optimize_hyp()
    # Train the best model with best hyperparameters
    run_id = train_best_model(best_params=best_params)
    # Specify the MLflow model URI
    #model_uri = f"runs:/a4519e1f7e00433886646a2bfb51600f/best_random_forest_model"
    # Sample data for inference (for the Iris dataset)
    data = [[5.1, 3.5, 1.4, 0.2]]
    # Perform inference using the ray_inference task
    inference_results , iris_type = ray_inference(model_uri=run_id,data=data)

    return best_accuracy, inference_results, iris_type

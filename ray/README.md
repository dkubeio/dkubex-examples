# Ray Training 

- Creating Ray Cluster 
    - With CPU
        ```
        $ d3x ray create -n cpu --cpu=4 --memory=8 --hcpu=4 --hmemory=8 
        ```
    - With GPU
        ```
        $ d3x ray create -n gpu --cpu=4 --memory=8 --hcpu=4 --hmemory=8 --gpu=1
        ```
- Activating Ray cluster for Training 
    ```
    $ d3x ray activate <cluster_name>
    ```
    Example
    ```
    $ d3x ray activate gpu
    ```
- Submitting training job to the ray cluster
    - Submit to the CPU cluster
        ```
        $ d3x ray job submit --submission-id ray-air --working-dir $PWD  --runtime-env-json='{"pip": ["torch==2.0.1", "torchvision"]}' -- python torch_fashion_mnist_example.py
        ```
    - Submit to the GPU cluster
        ```
        d3x ray job submit --submission-id ray-air --working-dir $PWD --runtime-env-json='{"pip": ["torch==2.0.1", "torchvision"]}' -- python torch_fashion_mnist_example.py --use-gpu --num-workers 1
        ```

- Viewing Ray job logs 
    - CLI
        ```
        $ d3x ray job logs <submission id>
        ```
    - UI
        for getting job logs in ui do the following
        - Open dkubex web ui
        - Go to clusters  tab
        - Click on the cluster 
        - In Ray dashboard go to jobs tab
        - Select the job based on the submission id

- MlFlow Correlation with Ray Jobs
    - Getting job id using submission id
        ```
        $ d3x ray job list | grep  <submission id>
        ```
        The results will have the job_id. 
    - Filtering MlFlow Runs using job id
        Filter MlFlow runs using the following search option. Replace '03000000' with your job id.
        ```
        tags.job_id = '03000000'
        ```
    **Note**:
        You could also use unique names for experiment & run

- Registering the model
    For registering the model in mlflow, do the following
    - Open dkubex web ui
    - Go to mlflow tab
    - Select the experiment name
    - Select the run, the run can be filtered using  tags.job_id = '{job_id}' and user_id = '{user}' and tags.`ray cluster` = '{cluster}'
    - Open the run 
    - Click on the register model button
    - Select the model (create if needed)
    - Submit 



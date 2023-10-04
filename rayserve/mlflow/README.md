# ML Flow deployments


- Fashion mnist model with GPU
  
    - Deploying using private repo
        ```
        $ d3x serve create -n <name> -r mlflow --model <registered model name> --model_version <model version>  --repo_name <repo name> --username <username> --is_private_repo --access_token <personal access token> --branch_name <branch> --depfilepath <deployment filepath> --ngpus <number of gpus>
        ```
        ```
        Example:
        
        $ d3x serve create -n fashion-mnist-deploy -r mlflow --model mnist --model_version 1 --is_private_repo --access_token XXXX --repo_name dkubex-examples --username dkubeio --branch_name mlflow --depfilepath fashion_mnist_gpu.deploy --ngpus 1
        ```

    - Deploying from local directory
        ```
        Example:

        $ d3x serve create -n fashion-mnist-deploy -r mlflow --model mnist --model_version 1 --depfilepath fashion_mnist_gpu.deploy --ngpus 1
        ```
        

- Fashion mnist model with cpu
  
    - Deploying using private repo
        ```
        Example:

        $ d3x serve create -n fashion-mnist-deploy -r mlflow --model mnist --model_version 1 --is_private_repo --access_token XXXX --repo_name dkubex-examples --username dkubeio --branch_name mlflow --depfilepath fashion_mnist_gpu.deploy
        ```

    - Deploying from local (user workspace)
        ```
        Example:
        
        $ d3x serve create -n fashion-mnist-deploy -r mlflow --model mnist --model_version 1 --depfilepath fashion_mnist_gpu.deploy --ngpus 1
        ```

- **Note**:
    - If you are trying out example from this repo
      - Make sure you are pointing to 'mlflow' branch, if you are 'deploying using private repo'.
      - Make sure you are inside MLFLOW directory for 'deploying from local'.
    - For public repo, ignore the options --is_private_repo & --access_token from 'deploying using private repo'

- **client** folder contains a python test code for inference test
   - fmnist_client.py contains instructions

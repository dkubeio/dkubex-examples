# Hugging Face deployments


- Hugging face model with GPU
  
    - Deploying using private repo
        ```
        $ d3x serve create -n <name> -r hugging_face --hface_repoid <repoid> --hface_tokenizer <tokenizer> --hface_classifier <classifier> --repo_name <repo name> --username <username> --is_private_repo --access_token <personal access token> --branch_name <branch> --depfilepath <deployment filepath> --ngpus <number of gpus>
        ```
        ```
        Example:
        
        $ d3x serve create -n hf-biogpt -r hugging_face --hface_repoid microsoft/biogpt --hface_tokenizer BioGptTokenizer --hface_classifier text-completion --repo_name dkubex-examples --username dkubeio --is_private_repo --access_token XXXXXX   --branch_name hugging_face --depfilepath biogpt.deploy --ngpus 1
        ```

    - Deploying from local directory
        ```
        Example:

        $ d3x serve create -n hf-biogpt -r hugging_face --hface_repoid microsoft/biogpt --hface_tokenizer BioGptTokenizer --hface_classifier text-completion --depfilepath biogpt.deploy --ngpus 1
        ```
        

- Hugging face model with cpu
  
    - Deploying using private repo
        ```
        Example:

        $ d3x serve create -n hf-biogpt -r hugging_face --hface_repoid microsoft/biogpt --hface_tokenizer BioGptTokenizer --hface_classifier text-completion --repo_name dkubex-examples --username dkubeio --is_private_repo --access_token XXXXXXXX   --branch_name hugging_face --depfilepath biogpt.deploy
        ```

    - Deploying from local (user workspace)
        ```
        Example:
        
        $ d3x serve create -n hf-biogpt -r hugging_face --hface_repoid microsoft/biogpt --hface_tokenizer BioGptTokenizer --hface_classifier text-completion --depfilepath biogpt.deploy
        ```

- **Note**:
    - If you are trying out example from this repo
      - Make sure you are pointing to 'hugging_face' branch, if you are 'deploying using private repo'.
      - Make sure you are inside HF directory for 'deploying from local'.
    - For public repo, ignore the options --is_private_repo & --access_token from 'deploying using private repo'

- **client** folder contains a python test code for inference test
   - hf_client.py contains instructions

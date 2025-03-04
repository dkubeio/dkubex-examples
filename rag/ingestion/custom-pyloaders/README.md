# Using Custom Dataloaders

Some data loaders require the python object of specific class as an input argument. For example Github repo loaders needs `GithubClient()` object.

In order to handle such special case, the user needs to write the loader function in separate python file as explained in below code snippet. The path/name of this file should be passed to pyloader parameter in [ingest.yaml](../ingest.yaml) file.

```
from pathlib import Path
from llama_index import download_loader
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from document_loaders import document_loader_func

# This is mandatory decorator to be used over loader function.
@document_loader_func(name="GitHubLoader", description="loader to load pdf, html & eml files with unstructured reader")
def GitHubLoader(inputs, reader):
    GithubRepositoryReader = download_loader('GithubRepositoryReader', custom_path="/tmp")

    loader_args = inputs.get("loader_args", {})
    data_args = inputs.get("data_args", {})

    github_client = GithubClient(loader_args['github_token'])
    loader = GithubRepositoryReader(
    github_client,
    owner =                  loader_args['owner'],
    repo =                   loader_args['repo'],
    verbose =                loader_args['verbose'],
    concurrent_requests =    loaders_args['concurrent_requests'],
    timeout =                loaders_args['timeout'],
    )

    documents = loader.load_data(**data_args)
    # alternatively, load from a specific commit:
    # docs = loader.load_data(commit_sha="a6c89159bf8e7086bea2f4305cff3f0a4102e370")
    return documents
```
## Here is an example file for the github loader. 
  User needs to generate GitHub Personal Access Token, by following below steps.
On GitHub acc => settings -> Developer Settings ->  Personal Access TOken -> Tokens(classic)

```
reader:
  - source: github_repo
    description: https://llamahub.ai/l/file?from=loaders
    inputs:
      loader_args:
        github_token: <User provided GitHub Personal Access Token>
        owner: oneconvergence
        repo: dkubex-fm
        include_dir:
          - tools
        include_file_ext:
          - .json
      data_args:
        branch: data_loader_arg
        #commit_sha: "a6c89159bf8e7086bea2f4305cff3f0a4102e370"
    pyloader: ./document_loaders/github.py
```
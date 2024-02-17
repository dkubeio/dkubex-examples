# How to use different Data Loaders (Data Readers)

Data Loaders are the very first step in ingestion pipeline. For different kind of source data, different data loaders are used. The complete list of these loaders are given at https://llamahub.ai/?tab=loaders. 

Use of these data-loaders are specified on [ingest.yaml](../ingest.yaml) file in the reader section at bottom of the file. Input parameters used in this section are explained here.

- **source**

    `source` indicates the type of data loader. Please visit https://llamahub.ai/?tab=loaders for all types of data loaders.

    *Example:*

    For wikipedia loader (https://llamahub.ai/l/wikipedia?from=loaders), take the title from `/wikipedia` as source parameter.

- **Loader arguments (`loader_args`)**

    These input parameters are used to create the object of specific reader(data loader).
    - Let's take example of `file` data-loader which creates object of a class- `SimpleDirectoryReader()`. 
    
        Input to this reader are:
        1. `input_dir`: type- `string`; Local directory path for source data corpus.
        2. `recursive`: type- `boolean`; `true` if the input data-corpus directory should be read recursively.
        3. `exclude_hidden`: type- `boolean`; `true` if the hidden file should not be loaded in ingestion dataset.

- **Data arguments (`data_args`):**

    These input parameters are used in `data_load` function. For example in case of wikipedia page loader, these arguments would be list of wikipedia-pages.

    ```
    pages:
      - 'Berlin'
      - 'Rome'
    ```

- **Object as an input argument (Special case):**

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

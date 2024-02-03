
# GRPC USAGE FOR DEV ACCOUNT

## STEPS 
### Installation and configuration of Flyte 
1. Install flytectl and flytekit on your local system using the following command 
__flytekit__ : *pip install flytekit==1.10.0*

__flytekit-plugins__ : *flytekitplugins-deck-standard*

__flytectl(mac)__ : *brew install flyteorg/homebrew-tap/flytectl*

__flytectl(linux)__ : *curl -sL https://ctl.flyte.org/install | bash*

__flytectl(upgrade)__ : *flytectl upgrade*

2. Intialize the flytectl config file by using the following command:
   *__flytectl config init__*
   
   This generates a *__config.yaml__* file in flyte directory which is usually in the path *__~/.flyte/config.yaml__*


3. Replace the config.yaml from the step2 with the config.yaml from the current directory of this repository.

### Workflow for running flyte remotely

Once the installation and configuration is done you can use the flytectl and pyflyte commands to create projects and run workflows and tasks.
1.  __Create Project__ : *__flytectl create project --name <name_of_the_project> --id <unique_id> --description "description" --labels app=flyte__* 

example : *flytectl create project --name flytesnacks --id flytesnacks --description "flytesnacks description" --labels app=flyte*

2. Use the mnist in dkubex-examples/flyte-examples folder from this repo and run the following command to execute a workflow remotely from your local system.

*__pyflyte run --remote --image <ecr_repo_image> mnist mnist_workflow__*

For the above example please use the Dockerfile inside the dkubex-examples/flyte-examples folder to build a docker image and push it to ecr repository and mention it in the --image args in the above command.

3. Once step 2 is successfully executed you can view your execution using flyte ui on dkubex.

 

*__NOTE__* : 

Make sure you are connected to the vpn as well as make sure you replace the config.yaml in the /.flyte folder in your local system with config.yaml from the current directory in order to connect remotely from your local system.


               
           
 




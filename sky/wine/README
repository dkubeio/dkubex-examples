Edit wine.yaml and specify MlFlow tracking URL

# To check for resources
d3x sky launch --env MLFLOW_TRACKING_TOKEN=$APIKEY -n wineft wine.yaml
# GPU node
d3x sky launch --env MLFLOW_TRACKING_TOKEN=$APIKEY -n wineft gpu.yaml

# To run a job
d3x sky launch -y --env MLFLOW_TRACKING_TOKEN=$APIKEY -n wineft wine.yaml

# This will look for the on-demand cheapest available resource and will start running the job 
# when it finds it . This basically reserves the VM and we need to terminate it on our own. 
d3x sky launch -y --env MLFLOW_TRACKING_TOKEN=$APIKEY -n wine-1 wine.yaml 


# This will look for the spot instances in every zone in the regions and will start running the job when it will get the cheapest available one 
# But the VMs could be prempted
d3x sky launch -y --env --use-spot MLFLOW_TRACKING_TOKEN=$APIKEY -n wine-1 wine.yaml


# Managed spot jobs helps us to automatically recover from preemptions. 
# If your spot VM is preempted (like someone else renting it), SkyPilot will automatically restart your job on another VM.
# Checkpointing needs to be done by User 
d3x sky launch spot -y --env MLFLOW_TRACKING_TOKEN=$APIKEY -n wine-1 wine.yaml


# Other useful commands
d3x sky check
d3x sky status --refresh
d3x sky down <cluster> -y  # deletes a cluster
d3x sky spot cancel -a -y  # cancels a job on spot instances

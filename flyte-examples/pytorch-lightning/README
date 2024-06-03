To run this example, we need dkubex with kubeflow and flyte enabled.
This example requires the pytorch plugin enabled in flyte and also gang scheduling enabled in kubeflow training operator.

Build docker image
------------------
docker build -t <repo>:<tag> -f mnist-lightning-dist.dockerfile .
docker push -t <repo>:<tag>

Run the workflow
----------------
Update NUM_NODES, NUM_DEVICES and CONTAINER_IMAGE in the mnist-lightning-dist.py and run
pyflyte run --remote mnist-lightning-dist.py pl_training_wf

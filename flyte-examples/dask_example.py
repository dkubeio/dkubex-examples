from flytekit import Resources, task
from flytekitplugins.dask import Dask
from dask import array as da
from flytekitplugins.dask import Dask, Scheduler, WorkerGroup
from flytekit import ImageSpec, Resources, task, workflow

custom_image = ImageSpec(name="flyte-dask-plugin", registry="ghcr.io/flyteorg", packages=["flytekitplugins-dask"])
if custom_image.is_container():
    from dask import array as da
    from flytekitplugins.dask import Dask
    from flytekitplugins.dask import Dask, Scheduler, WorkerGroup
@task(
  task_config=Dask(
      scheduler=Scheduler(
          limits=Resources(cpu="1", mem="500Mi"),  # Applied to the job pod
      ),
      workers=WorkerGroup(
          limits=Resources(cpu="1", mem="500Mi"), # Applied to the scheduler and worker pods
      ),
  ),
  container_image=custom_image
)

def hello_dask(size: int) -> float:
    # Dask will automatically generate a client in the background using the Client() function.
    # When running remotely, the Client() function will utilize the deployed Dask cluster.
    array = da.random.random(size)
    return float(array.mean().compute())
#if __name__ == "__main__":
    #print(hello_dask(size=1000))


#def my_dask_task() -> int:
    #dask creates a local client here
   #array = da.ones((1000,1000,1000))
    #return int(array.mean().compute())


@workflow
def dask_workflow() -> float:
    return hello_dask(size=1000)


from flytekit import Resources, task, workflow, Labels, Annotations, ImageSpec
from flytekitplugins.dask import Dask
from dask import array as da
'''
custom_image = ImageSpec(
    name="dask-flyte-plugin",
    registry="ghcr.io/flyteorg",
    packages=["flytekitplugins-dask","dask[complete]","flytekitplugins-envd"],
)

if custom_image.is_container():
    from flytekitplugins.dask import Dask
    import dask as da
'''
@task(requests=Resources(mem="1Gi", cpu="1"),task_config=Dask(),container_image="dkubex123/my_flyte_image:dask")
def my_dask_task():
    array = da.random.random(10)
    a = float(array.mean().compute())
    print(a)
    return a

@workflow
def my_dask_workflow():
   return my_dask_task()

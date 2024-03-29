import typing

from flytekit import ImageSpec, Resources, task, workflow

custom_image = ImageSpec(
    name="ray-flyte-plugin",
    registry="ghcr.io/flyteorg",
    packages=["flytekitplugins-ray"],
)

if custom_image.is_container():
    import ray
    from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

@ray.remote
def f(x):
    return x * x

#@task(
#    task_config=RayJobConfig(address="ray://flyteray-head-svc.ocdlgit:10001",runtime_env={"pip": ["numpy", "pandas"]})
#)
#def ray_task() -> typing.List[int]:
    #futures = [f.remote(i) for i in range(5)]
    #return ray.get(futures)

@task(task_config=RayJobConfig(worker_node_config=[WorkerNodeConfig(group_name="test-group", replicas=1)]),requests=Resources(cpu="1", mem="1Gi"),container_image=custom_image)
def ray_task() -> typing.List[int]:
    futures = [f.remote(i) for i in range(5)]
    return ray.get(futures)

@workflow
def ray_workflow(n: int) -> typing.List[int]:
    return ray_task()


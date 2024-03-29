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

ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=1)],
    runtime_env={"pip": ["numpy", "pandas"]},  # or runtime_env="./requirements.txt"
)

@task(
    task_config=ray_config,
    requests=Resources(mem="2Gi", cpu="2"),
    container_image=custom_image,
)
def ray_task(n: int) -> typing.List[int]:
    futures = [f.remote(i) for i in range(n)]
    return ray.get(futures)

@workflow
def ray_workflow(n: int) -> typing.List[int]:
    return ray_task(n=n)

if __name__ == "__main__":
    print(ray_workflow(n=10))

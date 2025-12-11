import ray
from continuous_runner_min import Runner

if __name__ == "__main__":
    ray.init()  # 集群用 ray.init(address="auto")
    runner = Runner.options(name="pipeline_runner").remote()
    runner.run.remote(script="./run_qwen2p5_pipeline.sh", log_dir="logs", backoff_s=5)
    print("Started. Actor name = pipeline_runner")

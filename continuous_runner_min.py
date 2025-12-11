import os, time, subprocess, ray

@ray.remote
class Runner:
    def run(self, script="./run_qwen2p5_pipeline.sh", log_dir="logs", backoff_s=5):
        os.makedirs(log_dir, exist_ok=True)
        while True:
            stamp = time.strftime("%m%d_%H%M%S")
            outlog = os.path.join(log_dir, f"pipeline_{stamp}.log")
            with open(outlog, "ab", buffering=0) as f:
                f.write(f"[{time.ctime()}] start: {script}\n".encode())
                p = subprocess.Popen(
                    ["bash", "-lc", script],
                    stdout=f, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # 方便优雅终止
                )
                with open(os.path.join(log_dir, "pipeline.pid"), "w") as pf:
                    pf.write(str(p.pid) + "\n")
                rc = p.wait()
                f.write(f"\n[{time.ctime()}] exit rc={rc}; restart in {backoff_s}s\n".encode())
            time.sleep(backoff_s)

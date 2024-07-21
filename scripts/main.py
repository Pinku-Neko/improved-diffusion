import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# List of tuples where each tuple contains the script name and its arguments
script = "scripts/image_fast_sample.py"
tolerance = 0
use_ddim = False
standard_args = ["--diffusion_steps", "4000", "--batch_size", "100", "--timestep_respacing", "1000", "--num_samples", "50000", "--use_ddim", f"{use_ddim}", "--tolerance", f"{tolerance}"]

# Generate cut_off values from 0 to 1 with 0.05 intervals
cut_off_values = [round(x * 0.05, 2) for x in range(21)]
scripts = [(script, standard_args + ["--cut_off", f"{cut_off}"]) for cut_off in cut_off_values]

def run_script(script, args, gpu_num):
    # Construct the command to run the script with its arguments
    command = ["python", script] + args
    # Set CUDA_VISIBLE_DEVICES environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    subprocess.run(command, env=env)

# Using ThreadPoolExecutor to run tasks concurrently
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for idx, (script, args) in enumerate(scripts):
        # Select GPU number based on index (0 or 1)
        gpu_num = idx % 2
        # Submit the task to the executor
        futures.append(executor.submit(run_script, script, args, gpu_num))

    # Wait for all tasks to complete
    for future in futures:
        future.result()

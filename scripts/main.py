import subprocess
import os

# List of tuples where each tuple contains the script name and its arguments
script = "scripts/image_fast_sample.py"
tolerance = 8
use_ddim = False
gpu_num = 1
standard_args = ["--diffusion_steps", "1000", "--batch_size", "128", "--timestep_respacing", "50", "--num_samples", "50000", "--use_ddim", f"{use_ddim}", "--tolerance", f"{tolerance}"]
scripts = [(script, standard_args+["--cut_off", "1.0"]), 
           (script, standard_args+["--cut_off", "0.9"]), 
           (script, standard_args+["--cut_off", "0.8"]), 
           (script, standard_args+["--cut_off", "0.7"]), 
           (script, standard_args+["--cut_off", "0.6"]), 
           (script, standard_args+["--cut_off", "0.5"]), 
           (script, standard_args+["--cut_off", "0.4"]), 
           (script, standard_args+["--cut_off", "0.3"]), 
           (script, standard_args+["--cut_off", "0.2"]), 
           (script, standard_args+["--cut_off", "0.1"]), 
           (script, standard_args+["--cut_off", "0.0"])]

# Iterate over the list and run each script with its arguments
for script, args in scripts:
    # Construct the command to run the script with its arguments
    command = ["python", script] + args
    # Set CUDA_VISIBLE_DEVICES environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    subprocess.run(command, env=env)


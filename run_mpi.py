#!/usr/bin/env python3
import subprocess
import os

def run_distributed():
    # Create correct hostfile
    hostfile_content = """# MLX Distributed Hostfile
localhost slots=1
mm@10.85.35.29 slots=1
mm@10.85.35.205 slots=1
"""
    
    with open('hostfile_corrected', 'w') as f:
        f.write(hostfile_content)
    
    # Run MPI
    cmd = [
        "mpirun", "-np", "3", 
        "--hostfile", "hostfile_corrected",
        "--mca", "btl_tcp_if_include", "en0",
        "/opt/homebrew/bin/python3.11", "distributed_inference.py"
    ]
    
    print("Running:", ' '.join(cmd))
    subprocess.run(cmd)
    
    # Cleanup
    os.remove('hostfile_corrected')

if __name__ == "__main__":
    run_distributed()
#!/usr/bin/env python3
import subprocess
import os
import sys
import json

class ClusterSetup:
    def __init__(self):
        self.nodes = [
            {"ip": "10.85.10.239", "name": "macbook-pro", "role": "master", "user": "zz"},
            {"ip": "10.85.35.29", "name": "mac-mini-1", "role": "worker", "user": "mm"},
            {"ip": "10.85.35.205", "name": "mac-mini-2", "role": "worker", "user": "mm"}
        ]
        
    def check_ssh(self):
        """Check SSH connectivity to all nodes"""
        print("🔌 Checking SSH connectivity...")
        all_good = True
        
        for node in self.nodes:
            if node["role"] == "master":
                continue  # Skip localhost
                
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", 
                 f"{node['user']}@{node['ip']}", "echo", "connected"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {node['name']} ({node['user']}@{node['ip']}): Connected")
            else:
                print(f"❌ {node['name']} ({node['user']}@{node['ip']}): Failed")
                all_good = False
                
        return all_good
    
    def setup_ssh(self):
        """Setup SSH keys for passwordless access"""
        print("\n🔐 Setting up SSH keys...")
        
        # Check if key exists
        key_path = os.path.expanduser("~/.ssh/id_ed25519")
        if not os.path.exists(key_path):
            print("Generating SSH key...")
            subprocess.run([
                "ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", ""
            ])
        
        # Copy to workers with correct username
        for node in self.nodes:
            if node["role"] == "worker":
                print(f"\nCopying SSH key to {node['name']}...")
                print(f"You may need to enter the password for {node['user']}@{node['ip']}")
                subprocess.run([
                    "ssh-copy-id", "-i", f"{key_path}.pub", 
                    f"{node['user']}@{node['ip']}"
                ])
#!/usr/bin/env python3
"""
Test script to verify ASR Fusion functionality
"""

import os
import sys
import time
import subprocess
import requests
import threading

def test_server_startup():
    """Test if the server starts correctly"""
    print("Testing server startup...")
    
    # Start the server in a separate process
    server_process = subprocess.Popen([
        sys.executable, "main.py", "--host", "localhost", "--port", "8603"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for the server to start
    time.sleep(5)
    
    try:
        # Check if server is responding
        response = requests.get("http://localhost:8603/health")
        if response.status_code == 200:
            print("✓ Server started successfully")
            print("✓ Health check passed")
            return server_process
        else:
            print("✗ Health check failed")
            return None
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server")
        return None

def test_api_endpoints():
    """Test API endpoints"""
    print("\nTesting API endpoints...")
    
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8603/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
        else:
            print("✗ Root endpoint failed")
            
        # Test health endpoint
        response = requests.get("http://localhost:8603/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print("✗ Health endpoint failed")
            
    except Exception as e:
        print(f"✗ API endpoint test failed: {e}")

def main():
    print("ASR Fusion Test Suite")
    print("=" * 30)
    
    # Check if audio file exists
    audio_file = "record_sys_250725_143258.m4a"
    if not os.path.exists(audio_file):
        print(f"Warning: Test audio file '{audio_file}' not found")
        print("Some tests will be skipped")
    
    # Test server startup
    server_process = test_server_startup()
    
    if server_process:
        # Test API endpoints
        test_api_endpoints()
        
        # Terminate the server
        server_process.terminate()
        server_process.wait()
        print("\nServer stopped")
    else:
        print("Server failed to start, skipping API tests")

if __name__ == "__main__":
    main()

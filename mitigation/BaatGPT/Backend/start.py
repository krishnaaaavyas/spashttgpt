import subprocess
import sys
import time
import os

def start_services():
    try:
        print("🚀 Starting ML Service...")
        # Start ML service
        ml_process = subprocess.Popen([
            sys.executable, 'ml_service.py'
        ], cwd='Backend')
        
        # Wait for ML service to start
        time.sleep(5)
        
        print("🚀 Starting Express Server...")
        # Start Node.js server
        node_process = subprocess.Popen([
            'npm', 'start'
        ], cwd='Backend')
        
        print("✅ Both services started!")
        print("Express API: http://localhost:5000")
        print("ML Service: http://localhost:5001")
        print("\nPress Ctrl+C to stop services")
        
        # Keep both running
        try:
            ml_process.wait()
            node_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            ml_process.terminate()
            node_process.terminate()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_services()

import os
import sys
import subprocess
import platform
from dotenv import load_dotenv

def start_server(port):
    os_name = platform.system().lower()
    
    print(f"🚀 Starting NaftaEC Chatbot on {os_name.capitalize()} (Port {port})...")
    
    if os_name == "windows":
        try:
            from waitress import serve
            from app import app
            print("Using Waitress (Production Server for Windows)")
            serve(app, host='0.0.0.0', port=port)
        except ImportError:
            print("Waitress not found. Falling back to Flask development server...")
            from app import app
            app.run(host='0.0.0.0', port=port)
    else:
        # Linux/macOS
        try:
            # Try to run with Gunicorn
            print("Using Gunicorn (Production Server for Unix)")
            cmd = ["gunicorn", "--workers", "4", "--bind", f"0.0.0.0:{port}", "app:app"]
            subprocess.run(cmd)
        except FileNotFoundError:
            print("Gunicorn not found. Falling back to Flask development server...")
            from app import app
            app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", 5001))
    start_server(port)

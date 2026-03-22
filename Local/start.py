import os
import sys
import subprocess
import platform
from dotenv import load_dotenv

def start_server(port):
    os_name = platform.system().lower()
    
    if os_name == "darwin":
        # Critical fix for Gunicorn on macOS + HuggingFace/Google SDKs
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        print("[OS] Darwin detected: Disabling OBJC Fork Safety for worker stability.")
    
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
            default_workers = 1
            workers = int(os.getenv("WEB_CONCURRENCY", str(default_workers)))
            max_requests = os.getenv("GUNICORN_MAX_REQUESTS", "1000")
            max_requests_jitter = os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100")
            timeout = os.getenv("GUNICORN_TIMEOUT", "60")
            print(f"Using Gunicorn (Production Server for Unix) with {workers} workers")
            cmd = [
                "gunicorn",
                "--workers", str(workers),
                "--bind", f"0.0.0.0:{port}",
                "--max-requests", str(max_requests),
                "--max-requests-jitter", str(max_requests_jitter),
                "--timeout", str(timeout),
                "app:app",
            ]
            subprocess.run(cmd)
        except FileNotFoundError:
            print("Gunicorn not found. Falling back to Flask development server...")
            from app import app
            app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", 5001))
    start_server(port)

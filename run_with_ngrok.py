import os
import sys
from pyngrok import ngrok, exception
from app import app  # Import your Flask app

def run_with_ngrok():
    try:
        # Set up ngrok
        port = 5000
        
        # Check if flask-cors is installed
        try:
            import flask_cors
        except ImportError:
            print("Error: flask-cors is not installed. Installing it now...")
            os.system("pip install flask-cors")
            print("Please restart this script.")
            return

        print("* Starting ngrok tunnel...")
        try:
            # Connect to ngrok
            public_url = ngrok.connect(port, bind_tls=True).public_url
            print(f"* ngrok tunnel \"{public_url}\" -> http://127.0.0.1:{port}")
            
            # Update the Flask app
            app.config["BASE_URL"] = public_url
            
            # Instructions for frontend
            print("\n" + "="*60)
            print("INTEGRATION INSTRUCTIONS")
            print("="*60)
            print(f"1. Update your Replit frontend to use this URL as the API endpoint:")
            print(f"   {public_url}")
            print("\n2. In your JavaScript code, replace:")
            print("   const apiUrl = \"http://localhost:5000\";")
            print("   with:")
            print(f"   const apiUrl = \"{public_url}\";")
            print("\n3. Your predict endpoint is available at:")
            print(f"   {public_url}/predict")
            print("="*60)
            
            # Run the app
            app.run(host="127.0.0.1", port=port, debug=False)
        
        except exception.PyngrokNgrokError as e:
            if "authtoken" in str(e):
                print("\nError: ngrok requires authentication!")
                print("\nFollow these steps:")
                print("1. Sign up for a free account at https://dashboard.ngrok.com/signup")
                print("2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken")
                print("3. Set up your auth token by running:")
                print("   python -c \"from pyngrok import ngrok; ngrok.set_auth_token('YOUR_AUTH_TOKEN')\"")
                print("\nThen run this script again.")
            else:
                print(f"\nError starting ngrok: {e}")
            
            return
    
    except KeyboardInterrupt:
        print("\nShutting down ngrok and Flask...")
        ngrok.kill()

if __name__ == "__main__":
    run_with_ngrok() 
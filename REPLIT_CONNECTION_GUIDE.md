# Connecting the Flask Backend to Replit Frontend

This guide will help you connect your local Flask backend to your Replit frontend.

## Option 1: Using ngrok (Recommended for Development)

Ngrok creates a secure tunnel to expose your localhost to the internet. This is the quickest way to get your backend accessible from your Replit frontend.

### Step 1: Create a free ngrok account

1. Go to [ngrok.com/signup](https://dashboard.ngrok.com/signup)
2. Create a free account
3. After signing up, go to the [setup page](https://dashboard.ngrok.com/get-started/setup)
4. Copy your authtoken (it looks like `1abc23defg45hijkl6m`)

### Step 2: Configure ngrok on your machine

Run the following command, replacing `YOUR_AUTH_TOKEN` with the token you copied:

```bash
# Using Python
python -c "from pyngrok import ngrok; ngrok.set_auth_token('YOUR_AUTH_TOKEN')"

# Or directly using the ngrok CLI if installed
# ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Step 3: Run Flask with ngrok

1. Make sure the Flask CORS extension is installed:
```bash
pip install flask-cors
```

2. Run your Flask app with ngrok:
```bash
python run_with_ngrok.py
```

3. You'll see output like this:
```
* ngrok tunnel "https://abc123def456.ngrok.app" -> http://127.0.0.1:5000
```

4. **Copy the ngrok URL** (https://abc123def456.ngrok.app in this example)

### Step 4: Update your Replit Frontend

1. Open your Replit frontend project
2. Update the API endpoint URL to the ngrok URL you copied
3. For example, change:
```javascript
const apiUrl = "http://localhost:5000";
```
to:
```javascript
const apiUrl = "https://abc123def456.ngrok.app";
```

4. Test your frontend to make sure it can connect to your backend

## Option 2: Deploy to a Cloud Service (For Production)

For a more permanent solution, you can deploy your Flask app to a cloud service:

### Popular Free/Low-Cost Options:

1. **Render**: [render.com](https://render.com/)
   - Free tier available
   - Relatively straightforward setup for Python/Flask apps

2. **Fly.io**: [fly.io](https://fly.io/)
   - Generous free tier
   - Good for small to medium apps

3. **Python Anywhere**: [pythonanywhere.com](https://www.pythonanywhere.com/)
   - Specific to Python applications
   - Free tier available

4. **Railway**: [railway.app](https://railway.app/)
   - Easy deployment from GitHub
   - Free tier for small projects

### Deployment Steps (General):

1. Create an account on your chosen platform
2. Connect your GitHub repository or upload your code
3. Configure the deployment settings for a Flask application
4. Deploy your application
5. Update your Replit frontend with the new backend URL

## Troubleshooting

### CORS Issues
If you encounter CORS issues, ensure that your Flask app has CORS properly configured:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes
```

### Testing the Connection
Use the `test_connection.html` file to verify your backend is accessible:

1. Open the file in a browser
2. Enter your backend URL (ngrok or deployed)
3. Select a video file and test the connection

### Ngrok Limitations
- Free ngrok URLs expire after a few hours
- You'll need to restart ngrok and update your Replit frontend URL when this happens
- For persistent development, consider a paid ngrok plan or deploy to a cloud service 
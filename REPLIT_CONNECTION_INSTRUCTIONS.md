# Connecting Flask Backend to Replit Frontend

This guide explains how to connect your local Flask backend to a frontend hosted on Replit.

## Option 1: Using ngrok (Temporary Public URL)

1. **Install ngrok** (if not already installed)
   ```bash
   # On Mac with Homebrew
   brew install ngrok
   
   # Or download from ngrok.com and install manually
   ```

2. **Sign up for ngrok** at https://ngrok.com/ and get your auth token

3. **Configure ngrok**
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

4. **Run your Flask backend**
   ```bash
   source .venv/bin/activate
   python app.py
   ```
   
5. **In a separate terminal, start ngrok to expose your backend**
   ```bash
   ngrok http 5000
   ```

6. **Copy the ngrok URL** (it will look like `https://a1b2c3d4.ngrok.io`)

7. **Configure your Replit frontend** to use this URL for API calls
   - Replace any API endpoints in your frontend code with the ngrok URL
   - For example: `fetch('https://a1b2c3d4.ngrok.io/predict', ...)`

Note: The ngrok URL will change every time you restart ngrok unless you have a paid plan.

## Option 2: Deploying to a Cloud Service (Permanent Solution)

For a more permanent solution, deploy your Flask backend to a cloud service:

1. **Choose a cloud platform**:
   - [Render](https://render.com/) (has a free tier)
   - [Railway](https://railway.app/)
   - [Heroku](https://www.heroku.com/) (limited free tier)
   - [PythonAnywhere](https://www.pythonanywhere.com/) (free tier available)

2. **Deploy your Flask application** following the platform's documentation

3. **Configure your Replit frontend** to use the permanent URL
   - Use the deployed URL in your frontend API calls

## Frontend Integration

In your Replit frontend, make API calls to your backend like this:

```javascript
// Example: Uploading a video file for prediction
async function predictFromVideo(videoFile) {
  const formData = new FormData();
  formData.append('file', videoFile);
  
  try {
    const response = await fetch('YOUR_BACKEND_URL/predict', {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

Replace `YOUR_BACKEND_URL` with either your ngrok URL or cloud service URL.

## Testing the Connection

To test if your connection is working:

1. Start your Flask backend
2. Expose it via ngrok or deploy it
3. Create a simple test in your Replit frontend to call the API
4. Monitor the Flask logs to see incoming requests

## Handling CORS

The Flask backend has been configured with CORS support to allow requests from any origin. If you need to restrict this to specific origins, modify the CORS configuration in `app.py`. 
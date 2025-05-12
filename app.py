import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model_utils import initialize_lipnet_model, predict_icu_phrase, ICU_PHRASES

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Lipreading API is running! Use /predict to make predictions."    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(video_path)
            app.logger.info(f"File saved to {video_path}")
            
            # Run lipreading prediction
            predicted_phrase = predict_icu_phrase(video_path)
            app.logger.info(f"Prediction for {filename}: {predicted_phrase}")
            
            # Clean up the uploaded file (optional)
            # os.remove(video_path)
            # app.logger.info(f"Cleaned up {video_path}")
            
            return jsonify({"prediction": predicted_phrase, "options": ICU_PHRASES}), 200
            
        except Exception as e:
            app.logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
            # Clean up if error occurs during processing
            if os.path.exists(video_path):
                 os.remove(video_path)
            return jsonify({"error": "Error processing file", "details": str(e)}), 500
    else:
        app.logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed. Only .mp4 files are accepted."}), 400

if __name__ == '__main__':
    app.logger.info("Initializing LipNet model (placeholder)...")
    initialize_lipnet_model() # Initialize model on startup (placeholder)
    app.logger.info("Starting Flask server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True) 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify, render_template
import os
import uuid
from grad_cam import get_gradcam_heatmap, overlay_heatmap
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from model_utils import check_image_quality, preprocess_image, predict_skin_condition
from database import save_screening, get_all_screenings, get_screening_by_id, get_last_screening_by_patient, init_db

app = Flask(__name__)

# ─── Load Grad-CAM Model (placeholder until you train your own) ───
gradcam_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# ─── Configuration ───
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Initialize Database ───
# This will create the database and table if they don't exist
init_db()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload, run quality check, return prediction, and save to database.
    """
    # 1. Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    # 2. Save the uploaded image with a unique name
    unique_id = str(uuid.uuid4())
    original_filename = file.filename
    safe_filename = f"{unique_id}_{original_filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)

    try:
        # 3. IMAGE QUALITY CHECK
        is_good, quality_msg = check_image_quality(filepath)

        if not is_good:
            # Delete the file since it failed quality check
            os.remove(filepath)
            return jsonify({
                'success': False,
                'quality_check': quality_msg,
                'message': 'Please upload a clearer, well-lit image.'
            })

        # 4. DUMMY PREDICTION (placeholder)
        prediction_result = predict_skin_condition(filepath)

        # 5. GENERATE GRAD-CAM HEATMAP
        gradcam_url = None
        gradcam_path = None
        try:
            # Preprocess the image for the Grad-CAM model
            img_array = preprocess_image(filepath)
            
            # Generate the heatmap
            heatmap = get_gradcam_heatmap(gradcam_model, img_array)
            
            # Overlay the heatmap on the original image and save it
            gradcam_output_path = overlay_heatmap(filepath, heatmap, alpha=0.5)
            
            # Convert to URL path (so the frontend can access it)
            gradcam_url = '/' + gradcam_output_path.replace('\\', '/')
            gradcam_path = gradcam_output_path
            
        except Exception as grad_cam_error:
            print(f"Grad-CAM warning: {grad_cam_error}")
            gradcam_url = None
            gradcam_path = None

        # 6. SAVE TO DATABASE
        patient_name = request.form.get('patient_name', 'Anonymous')
        
        record_id = save_screening(
            patient_name=patient_name,
            image_path=filepath,
            gradcam_path=gradcam_path,
            prediction=prediction_result['class'],
            confidence=prediction_result['confidence'],
            risk_level=prediction_result['risk'],
            recommendation=prediction_result['recommendation']
        )

        # 7. CHECK FOR TREND (Risk escalation)
        trend_message = None
        last_record = get_last_screening_by_patient(patient_name)
        if last_record:
            # If there is a previous record and current risk is higher than previous
            # (This compares "Lower Concern" vs "High Concern")
            if last_record['risk_level'] == 'Lower Concern' and prediction_result['risk'] == 'High Concern':
                trend_message = f"⚠️ Risk level has increased since your last screening on {last_record['timestamp']}. Please consult a dermatologist."
            elif last_record['risk_level'] == 'High Concern' and prediction_result['risk'] == 'Lower Concern':
                trend_message = f"✅ Risk level has decreased since your last screening on {last_record['timestamp']}. Continue monitoring."
            else:
                trend_message = f"📊 Your risk level is consistent with your last screening on {last_record['timestamp']}."

        # 8. FINAL RESPONSE
        return jsonify({
            'success': True,
            'quality_check': quality_msg,
            'prediction': prediction_result['class'],
            'confidence': prediction_result['confidence'],
            'risk_level': prediction_result['risk'],
            'recommendation': prediction_result['recommendation'],
            'gradcam_image': gradcam_url,
            'record_id': record_id,
            'trend_message': trend_message
        })

    except Exception as e:
        # Clean up the file if something goes wrong
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
def history():
    """Display all past screenings."""
    records = get_all_screenings(limit=50)
    return render_template('history.html', records=records)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
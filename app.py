import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template, Response
import uuid
from grad_cam import get_gradcam_heatmap, overlay_heatmap
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from model_utils import check_image_quality, preprocess_image, predict_skin_condition
from database import save_screening, get_all_screenings, get_screening_by_id, get_last_screening_by_patient, init_db

# ─── ReportLab imports ───
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

app = Flask(__name__)

# ─── Load Grad-CAM Model (placeholder until you train your own) ───
gradcam_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# ─── Configuration ───
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Initialize Database ───
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
            img_array = preprocess_image(filepath)
            heatmap = get_gradcam_heatmap(gradcam_model, img_array)
            gradcam_output_path = overlay_heatmap(filepath, heatmap, alpha=0.5)
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
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
def history():
    """Display all past screenings."""
    records = get_all_screenings(limit=50)
    return render_template('history.html', records=records)


@app.route('/download-pdf/<int:record_id>')
def download_pdf(record_id):
    """Generate and download a PDF report using ReportLab."""
    record = get_screening_by_id(record_id)
    if not record:
        return "Record not found", 404

    # Create BytesIO buffer
    buffer = BytesIO()

    # Create document
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    # ─── Styles ───
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=24,
        textColor=colors.HexColor('#0b2a44')
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=14,
        textColor=colors.HexColor('#4a6a85')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=6
    )
    
    normal_style = styles['Normal']

    # ─── Build story ───
    story = []

    # Title
    story.append(Paragraph("🩺 Meiyalagu AI", title_style))
    story.append(Paragraph("Intelligent Skin Disease & Skin Cancer Risk Screening", subtitle_style))
    story.append(Spacer(1, 12*mm))

    # Patient Info
    story.append(Paragraph(f"<b>Patient:</b> {record['patient_name']}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {record['timestamp']}", normal_style))
    story.append(Paragraph(f"<b>Report ID:</b> #{record['id']}", normal_style))
    story.append(Spacer(1, 8*mm))

    # ─── Images ───
    # Check if images exist and add them
    img_table_data = []

    # Original image
    if record['image_path'] and os.path.exists(record['image_path']):
        try:
            img_flow = Image(record['image_path'], width=70*mm, height=70*mm)
            img_table_data.append([Paragraph("<b>📷 Uploaded Image</b>", normal_style), img_flow])
        except Exception as e:
            print(f"Error loading image: {e}")

    # Grad-CAM image
    if record['gradcam_path'] and os.path.exists(record['gradcam_path']):
        try:
            img_flow = Image(record['gradcam_path'], width=70*mm, height=70*mm)
            img_table_data.append([Paragraph("<b>🔥 AI Attention (Grad-CAM)</b>", normal_style), img_flow])
        except Exception as e:
            print(f"Error loading Grad-CAM image: {e}")

    if img_table_data:
        img_table = Table(img_table_data, colWidths=[60*mm, 80*mm])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8*mm))

    # ─── Disease Classification ───
    story.append(Paragraph("<b><font color='#1a73e8'>🧬 Skin Disease Classification</font></b>", heading_style))
    story.append(Paragraph(f"<b>Prediction:</b> {record['prediction']}", normal_style))
    conf = record['confidence'] * 100
    story.append(Paragraph(f"<b>Confidence:</b> {conf:.0f}%", normal_style))
    story.append(Spacer(1, 4*mm))

    # ─── Cancer Risk Screening ───
    risk_color = '#dc3545' if record['risk_level'] == 'High Concern' else '#0c5460'
    story.append(Paragraph(f"<b><font color='{risk_color}'>🚨 Cancer Risk Screening</font></b>", heading_style))
    story.append(Paragraph(f"<b>Risk Level:</b> {record['risk_level']}", normal_style))
    story.append(Paragraph(f"<b>Recommendation:</b> {record['recommendation']}", normal_style))
    story.append(Spacer(1, 8*mm))

    # ─── Disclaimer ───
    story.append(Paragraph("<b>⚠️ Important Disclaimer:</b>", normal_style))
    story.append(Paragraph(
        "This is an AI-assisted screening tool and does NOT replace a clinical dermatologist. "
        "It is designed for educational and decision-support purposes only.",
        normal_style
    ))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f"Generated by Meiyalagu AI on {record['timestamp']}", normal_style))

    # ─── Build PDF ───
    doc.build(story)

    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()

    # Return as downloadable file
    response = Response(pdf_data, content_type='application/pdf')
    response.headers['Content-Disposition'] = f'attachment; filename=Meiyalagu_Report_{record_id}.pdf'
    return response


@app.route('/debug-routes')
def debug_routes():
    """Show all registered routes."""
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return "<br>".join(routes)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
import cv2
import numpy as np
from PIL import Image
import random

# --- 1. IMAGE QUALITY CHECK (Rule-based) ---
def check_image_quality(image_path):
    """
    Checks if the image is clear and well-lit using OpenCV.
    Returns: (is_good: bool, message: str)
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read the image file."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur detection (Laplacian variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness detection (mean pixel intensity)
    brightness = np.mean(gray)

    # Thresholds (tuned for skin lesion images)
    if blur_score < 100:
        return False, f"Image is too blurry (score: {blur_score:.0f}). Please retake."
    if brightness < 50:
        return False, f"Image is too dark (brightness: {brightness:.0f}). Please use better lighting."
    if brightness > 220:
        return False, f"Image is overexposed (brightness: {brightness:.0f}). Please adjust lighting."

    return True, f"Quality passed (Blur: {blur_score:.0f}, Brightness: {brightness:.0f})"


# --- 2. PREPROCESSING (for MobileNetV2) ---
def preprocess_image(image_path):
    """
    Loads the image, resizes to 224x224, and applies MobileNetV2 preprocessing.
    Returns: numpy array ready for model input.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 expects pixel values in [-1, 1] range
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)
    return img_array


# --- 3. DUMMY PREDICTION (PLACEHOLDER) ---
# This simulates a real AI prediction.
# You will replace this with your actual trained model later.

CLASS_NAMES = [
    'Melanocytic Nevus (nv)',
    'Melanoma (mel)',
    'Benign Keratosis (bkl)',
    'Basal Cell Carcinoma (bcc)',
    'Actinic Keratosis (akiec)',
    'Vascular Lesion (vasc)',
    'Dermatofibroma (df)'
]

# High-risk classes (we map these to "High Concern")
HIGH_RISK_SET = {'melanoma', 'bcc', 'akiec'}

def predict_skin_condition(image_path):
    """
    For NOW: Returns a random dummy prediction to test the pipeline.
    LATER: You will replace this with: model.predict(preprocessed_img)
    """
    # Seed random based on the file path for consistency
    random.seed(image_path)
    class_idx = random.randint(0, len(CLASS_NAMES) - 1)
    confidence = round(random.uniform(0.65, 0.98), 2)

    predicted_class = CLASS_NAMES[class_idx]

    # Check if this class is high-risk
    is_high_risk = any(key in predicted_class.lower() for key in HIGH_RISK_SET)

    return {
        "class": predicted_class,
        "confidence": confidence,
        "risk": "High Concern" if is_high_risk else "Lower Concern",
        "recommendation": (
            "⚠️ Professional dermatological evaluation is strongly recommended."
            if is_high_risk else
            "✅ Monitor the lesion. Consult a professional if it changes in size, shape, or color."
        )
    }
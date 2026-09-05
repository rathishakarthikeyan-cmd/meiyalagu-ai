import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap for the given image and model.
    
    Args:
        model: A Keras model.
        img_array: Preprocessed image array (1, 224, 224, 3).
        last_conv_layer_name: Name of the last conv layer (auto-detected if None).
    
    Returns:
        heatmap: 2D numpy array of shape (height, width) with values in [0, 1].
    """
    # Auto-detect the last convolutional layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name and len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
        else:
            raise ValueError("Could not find a convolutional layer.")
    else:
        last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create a model that maps input to conv layer output and final predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        # Use the top predicted class
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the predicted class with respect to conv output
    grads = tape.gradient(class_channel, conv_output)

    # Global average pooling over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output by the gradients
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(image_path, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and saves the result.
    
    Args:
        image_path: Path to the original image.
        heatmap: 2D numpy array from get_gradcam_heatmap().
        alpha: Transparency of the heatmap overlay (0.0 to 1.0).
    
    Returns:
        output_path: Path to the saved overlay image.
    """
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    # Convert heatmap to RGB colormap (JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    # ─── FIX: Save to static/ folder so Flask can serve it ───
    # Replace 'uploads' with 'static' in the path
    output_path = image_path.replace('uploads', 'static').replace('.', '_gradcam.')
    
    cv2.imwrite(output_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    return output_path
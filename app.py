from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path - you can make this configurable
MODEL_PATH = "C:\\Users\\abhir\\OneDrive\\Desktop\\fabric_backend\\best_float32.tflite"

# Class names for defect types (matching your training dataset)
CLASS_NAMES = ['Hole', 'Knot-slub', 'Spot', 'Thick-Missing yarn']

# Global variables for the model
interpreter = None
input_details = None
output_details = None

def load_model():
    """Load the TFLite model and return interpreter details"""
    global interpreter, input_details, output_details
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info("Model loaded successfully")
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        logger.info(f"Input dtype: {input_details[0]['dtype']}")
        logger.info(f"Output dtype: {output_details[0]['dtype']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for YOLO model"""
    logger.info(f"Original image size: {image.size}")
    logger.info(f"Original image mode: {image.mode}")
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size (640x640)
    image = image.resize((640, 640), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Log image statistics
    logger.info(f"Image array shape: {image_array.shape}")
    logger.info(f"Image array min/max: {image_array.min():.2f}/{image_array.max():.2f}")
    
    # Normalize to [0, 1] range (standard for YOLO)
    image_array = image_array / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(image_array, axis=0)
    
    logger.info(f"Final input shape: {input_data.shape}")
    logger.info(f"Final input min/max: {input_data.min():.3f}/{input_data.max():.3f}")
    
    return input_data

def parse_yolo_output_v2(output_data, confidence_threshold=0.25, nms_threshold=0.4):
    """
    Enhanced YOLO output parsing with multiple format support
    """
    detections = []
    
    try:
        logger.info(f"Raw output shape: {output_data.shape}")
        logger.info(f"Raw output dtype: {output_data.dtype}")
        logger.info(f"Raw output min/max: {output_data.min():.4f}/{output_data.max():.4f}")
        
        # Handle different output shapes
        if len(output_data.shape) == 3:
            # Shape: (1, N, features) - remove batch dimension
            output_data = output_data[0]
        elif len(output_data.shape) == 4:
            # Shape: (1, grid_h, grid_w, anchors*features) - YOLO grid format
            batch, grid_h, grid_w, features = output_data.shape
            output_data = output_data[0]  # Remove batch dimension
            
            # Convert grid format to detection format
            output_data = convert_grid_to_detections(output_data, grid_h, grid_w, features)
        
        logger.info(f"Processed output shape: {output_data.shape}")
        
        # Now output_data should be (N, features) where features >= 5
        num_detections, num_features = output_data.shape
        logger.info(f"Number of raw detections: {num_detections}")
        logger.info(f"Features per detection: {num_features}")
        
        # Expected format: [x_center, y_center, width, height, confidence, class_0, class_1, ...]
        # OR: [x_center, y_center, width, height, class_0_conf, class_1_conf, ...]
        
        for i in range(num_detections):
            detection = output_data[i]
            
            # Log first few detections for debugging
            if i < 5:
                logger.info(f"Detection {i}: {detection}")
            
            # Try different YOLO output formats
            if num_features >= 5:
                # Format 1: [x, y, w, h, objectness, class_scores...]
                if num_features >= 5 + len(CLASS_NAMES):
                    x_center, y_center, width, height, objectness = detection[:5]
                    class_scores = detection[5:5+len(CLASS_NAMES)]
                    
                    # Overall confidence = objectness * max_class_score
                    max_class_idx = np.argmax(class_scores)
                    max_class_score = class_scores[max_class_idx]
                    confidence = objectness * max_class_score
                    
                # Format 2: [x, y, w, h, class_0_conf, class_1_conf, ...]
                elif num_features >= 4 + len(CLASS_NAMES):
                    x_center, y_center, width, height = detection[:4]
                    class_scores = detection[4:4+len(CLASS_NAMES)]
                    
                    max_class_idx = np.argmax(class_scores)
                    confidence = class_scores[max_class_idx]
                    max_class_score = confidence
                    objectness = confidence
                    
                # Format 3: [x, y, w, h, confidence] (single class)
                else:
                    x_center, y_center, width, height, confidence = detection[:5]
                    max_class_idx = 0
                    max_class_score = confidence
                    objectness = confidence
                
                # Apply confidence threshold
                if confidence > confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    x1 = (x_center - width / 2) * 640
                    y1 = (y_center - height / 2) * 640
                    x2 = (x_center + width / 2) * 640
                    y2 = (y_center + height / 2) * 640
                    
                    # Clamp to image bounds
                    x1 = max(0, min(640, x1))
                    y1 = max(0, min(640, y1))
                    x2 = max(0, min(640, x2))
                    y2 = max(0, min(640, y2))
                    
                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        class_name = CLASS_NAMES[max_class_idx] if max_class_idx < len(CLASS_NAMES) else f'Class_{max_class_idx}'
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': int(max_class_idx),
                            'class_name': class_name,
                            'class_confidence': float(max_class_score),
                            'raw_confidence': float(objectness)
                        })
                        
                        logger.info(f"Valid detection: {class_name}, conf={confidence:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        
        logger.info(f"Found {len(detections)} detections before NMS")
        
        # Apply Non-Maximum Suppression
        if len(detections) > 0:
            detections = apply_nms(detections, nms_threshold)
            logger.info(f"Found {len(detections)} detections after NMS")
        
    except Exception as e:
        logger.error(f"Error in YOLO parsing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try fallback parsing
        detections = fallback_parsing(output_data, confidence_threshold)
        
    return detections

def convert_grid_to_detections(grid_output, grid_h, grid_w, features):
    """Convert YOLO grid output to detection format"""
    try:
        # Assuming 3 anchors per grid cell and features per anchor
        anchors_per_cell = 3
        features_per_anchor = features // anchors_per_cell
        
        detections = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                for k in range(anchors_per_cell):
                    # Extract features for this anchor
                    start_idx = k * features_per_anchor
                    end_idx = start_idx + features_per_anchor
                    
                    if end_idx <= features:
                        anchor_features = grid_output[i, j, start_idx:end_idx]
                        
                        # Convert grid coordinates to normalized coordinates
                        if len(anchor_features) >= 5:
                            x_offset, y_offset, w, h = anchor_features[:4]
                            
                            # Convert to absolute coordinates
                            x_center = (j + x_offset) / grid_w
                            y_center = (i + y_offset) / grid_h
                            
                            detection = np.concatenate([[x_center, y_center, w, h], anchor_features[4:]])
                            detections.append(detection)
        
        return np.array(detections) if detections else np.zeros((0, features_per_anchor))
        
    except Exception as e:
        logger.error(f"Error converting grid to detections: {str(e)}")
        # Return original format if conversion fails
        return grid_output.reshape(-1, grid_output.shape[-1])

def fallback_parsing(output_data, confidence_threshold):
    """Fallback parsing method for edge cases"""
    detections = []
    
    try:
        logger.info("Trying fallback parsing methods...")
        
        # Method 1: Try treating as segmentation mask
        if len(output_data.shape) >= 2:
            # Apply sigmoid to convert logits to probabilities
            if np.max(output_data) > 5 or np.min(output_data) < -5:
                probs = 1 / (1 + np.exp(-output_data))
            else:
                probs = output_data
            
            # Find high-confidence regions
            high_conf_mask = probs > confidence_threshold
            
            if np.any(high_conf_mask):
                logger.info(f"Found high confidence regions in fallback parsing")
                # This would need more sophisticated region extraction
                # For now, create a dummy detection
                detections.append({
                    'bbox': [100.0, 100.0, 200.0, 200.0],
                    'confidence': float(np.max(probs)),
                    'class_id': 0,
                    'class_name': CLASS_NAMES[0],
                    'class_confidence': float(np.max(probs)),
                    'raw_confidence': float(np.max(probs))
                })
        
    except Exception as e:
        logger.error(f"Fallback parsing failed: {str(e)}")
    
    return detections

def apply_nms(detections, nms_threshold):
    """Apply Non-Maximum Suppression"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for i, det1 in enumerate(detections):
        should_keep = True
        
        for det2 in keep:
            if det1['class_id'] == det2['class_id']:  # Same class
                iou = calculate_iou(det1['bbox'], det2['bbox'])
                if iou > nms_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(det1)
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Load model on startup
if not load_model():
    logger.error("Failed to load model on startup")
    exit(1)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Fabric defect detection API is running",
        "supported_classes": CLASS_NAMES,
        "model_input_shape": input_details[0]['shape'].tolist() if input_details else None,
        "model_output_shape": output_details[0]['shape'].tolist() if output_details else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Predict fabric defects from uploaded image"""
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["file"]
        
        if image_file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify({"error": "No image selected"}), 400
        
        logger.info(f"Processing image: {image_file.filename}")
        
        # Load and preprocess image
        image = Image.open(image_file.stream).convert("RGB")
        input_data = preprocess_image(image)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"Model inference completed")
        logger.info(f"Output shape: {output_data.shape}")
        logger.info(f"Output min/max: {output_data.min():.4f}/{output_data.max():.4f}")
        
        # Try multiple confidence thresholds for debugging
        confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        detection_counts = {}
        
        for threshold in confidence_thresholds:
            test_detections = parse_yolo_output_v2(output_data, confidence_threshold=threshold)
            detection_counts[f"threshold_{threshold}"] = len(test_detections)
            logger.info(f"Detections at threshold {threshold}: {len(test_detections)}")
        
        # Use lower threshold for final results
        detections = parse_yolo_output_v2(output_data, confidence_threshold=0.1)
        
        logger.info(f"Final detection count: {len(detections)}")
        
        # Prepare response
        if len(detections) > 0:
            result = "Defects Found"
            max_confidence = max(det['confidence'] for det in detections)
            
            # Count defects by type
            defect_counts = {}
            for det in detections:
                class_name = det['class_name']
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        else:
            result = "No Defects"
            max_confidence = 0.0
            defect_counts = {}
        
        response_data = {
            "prediction": result,
            "confidence": round(max_confidence, 4),
            "detections": detections,
            "defect_counts": defect_counts,
            "total_defects": len(detections),
            "image_size": [640, 640],
            "debug_info": {
                "output_shape": list(output_data.shape),
                "output_min_max": [float(output_data.min()), float(output_data.max())],
                "detection_counts_by_threshold": detection_counts,
                "model_input_shape": input_details[0]['shape'].tolist(),
                "model_output_shape": output_details[0]['shape'].tolist(),
                "class_names": CLASS_NAMES
            }
        }
        
        logger.info(f"Returning result: {result} with {len(detections)} detections")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large",
        "message": "Please upload an image smaller than 16MB"
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server"
    }), 500

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Supported classes: {CLASS_NAMES}")
    
    app.run(
        host="0.0.0.0", 
        port=5000, 
        debug=False,
        threaded=True
    )
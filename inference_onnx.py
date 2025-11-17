"""
Inference script for the Fake Image Detector ONNX model.
Demonstrates how to load and use the exported ONNX model for predictions.
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Load the ONNX model
onnx_model_path = "fake_image_detector.onnx"
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

print(f"Loading ONNX model from {onnx_model_path}...")
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    """
    Predict whether an image is fake or real using the ONNX model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: {"prediction": "Real" or "Fake", "confidence": float (0-1)}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    
    # Add batch dimension: (3, 128, 128) -> (1, 3, 128, 128)
    input_data = np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)
    
    # Run inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    output = ort_session.run([output_name], {input_name: input_data})[0]
    confidence = float(output[0][0])
    
    # Classify based on output (sigmoid output: 0-1)
    # Model trained with: Real=0, Fake=1
    prediction = "Fake" if confidence > 0.5 else "Real"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "raw_output": float(confidence)
    }

def batch_predict(image_dir, class_name="fake"):
    """
    Predict on all images in a directory.
    
    Args:
        image_dir: Path to directory containing images
        class_name: Class name (e.g., "fake", "real")
        
    Returns:
        dict: Summary statistics
    """
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Directory not found: {image_dir}")
    
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print(f"\nProcessing images in {image_dir}...")
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            filepath = os.path.join(image_dir, filename)
            try:
                result = predict_image(filepath)
                results.append({
                    "filename": filename,
                    **result
                })
                print(f"  {filename}: {result['prediction']} (confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"  {filename}: Error - {e}")
    
    # Summary
    if results:
        predictions = [r['prediction'] for r in results]
        fake_count = predictions.count('Fake')
        real_count = predictions.count('Real')
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return {
            "total_images": len(results),
            "fake_count": fake_count,
            "real_count": real_count,
            "average_confidence": avg_confidence,
            "results": results
        }
    return None

if __name__ == "__main__":
    # Example usage
    print("ONNX Model Inference Example")
    print("=" * 50)
    
    # Test on a single image (update path as needed)
    test_image = "data/test/fake/sample.jpg"
    
    if os.path.exists(test_image):
        print(f"\nPredicting on: {test_image}")
        result = predict_image(test_image)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Test image not found: {test_image}")
        print("To test, provide an image path and run:")
        print("  python -c \"from inference_onnx import predict_image; print(predict_image('path/to/image.jpg'))\"")
    
    # Batch prediction example (uncomment to use)
    # test_dir = "data/test/fake"
    # if os.path.isdir(test_dir):
    #     summary = batch_predict(test_dir)
    #     if summary:
    #         print(f"\nSummary: {summary['fake_count']} fake, {summary['real_count']} real")
    #         print(f"Average confidence: {summary['average_confidence']:.4f}")

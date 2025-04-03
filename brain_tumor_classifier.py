import numpy as np
from PIL import Image
import onnxruntime as ort

# Initialize ONNX Runtime session
session = ort.InferenceSession("brain_tumor_classifier.onnx")

def predict_image(image_path):
    """
    Preprocesses the input image, performs inference using ONNX Runtime,
    and returns the predicted class name and probability.
    """
    # Define the class labels
    classes = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    
    # Preprocessing steps:
    # 1. Open the image and ensure it's in RGB format.
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize the image to 224x224.
    image = image.resize((224, 224))
    
    # 3. Convert the image to a NumPy array and scale pixel values to [0, 1].
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # 4. Normalize the image using ImageNet mean and std.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # 5. Rearrange dimensions to channel-first format: [C, H, W].
    image_np = np.transpose(image_np, (2, 0, 1))
    
    # 6. Add a batch dimension: [1, C, H, W].
    image_np = np.expand_dims(image_np, axis=0)
    
    # Perform inference with ONNX Runtime.
    # Explicitly cast the input tensor to np.float32 to ensure compatibility.
    image_np = image_np.astype(np.float32)
    inputs = {session.get_inputs()[0].name: image_np}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    
    # Compute softmax probabilities.
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Get the predicted class index and corresponding label.
    class_idx = np.argmax(probabilities, axis=1)[0]
    class_name = classes[class_idx]
    prob = probabilities[0, class_idx] * 100
    
    return class_name, prob

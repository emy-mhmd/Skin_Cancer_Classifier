# Imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
CORS(app)

# Function to create models
def create_model(model_name):
    if model_name == 'resnet':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    elif model_name == 'efficientnetb5':
        model = EfficientNet.from_name('efficientnet-b5')  # Use the model name from efficientnet_pytorch

        # Replace the fully connected layer (classifier) to match your output
        model._fc = torch.nn.Sequential(
            torch.nn.Linear(model._fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),  # Adjust to your output size (1 for binary classification)
            torch.nn.Sigmoid()  # Sigmoid for binary classification
        )
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()  # For binary classification
    )

    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights=None)

    # Replace the classifier
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.last_channel, 128),  # Input size is model.last_channel (1280 for MobileNetV2)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),  # Binary classification output
            torch.nn.Sigmoid()       # For binary classification
        )
    else:
        raise ValueError("Invalid model name.")
    return model

# Function to load weights
def load_model_weights(model, model_name):
    checkpoint_path = f'./models/{model_name}.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint

    # Remove mismatched keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model

# Load all models
models_dict = {
    'resnet': load_model_weights(create_model('resnet'), 'resnet'),
    'efficientnetb5': load_model_weights(create_model('efficientnetb5'), 'efficientnetb5'),
    'vgg16': load_model_weights(create_model('vgg16'), 'vgg16'),
    'mobilenet': load_model_weights(create_model('mobilenet'), 'mobilenet'),
}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
   #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'File and model name are required'}), 400

    file = request.files['file']
    model_name = request.form['model']

    if model_name == 'default':
        # Use the ensemble method for 'default'
        result, confidence = ensemble_predict(file)
        return jsonify({'prediction': result, 'confidence': f"{confidence:.2f}%"})

    if model_name not in models_dict:
        return jsonify({'error': f'Model "{model_name}" not supported'}), 400

    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)

        model = models_dict[model_name]
        with torch.no_grad():
            output = model(img)
            pred = (output > 0.5).float().item()
            confidence = (output*100 ).float().item() # Confidence as percentage
            print(f"{confidence}%")

        result = 'Cancerous' if pred == 1 else 'Non-Cancerous'
        if result == 'Non-Cancerous':
            confidence=100-confidence
        if confidence >100:
            confidence = 100
        if confidence < 0:
            confidence = 0
        return jsonify({'prediction': result, 'confidence': f"{confidence:.2f}%"})

    return jsonify({'error': 'Invalid file'}), 400


# Function to load models and perform ensemble prediction
def ensemble_predict(file):
    # Prepare the image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = transform(img).unsqueeze(0)

    # Get predictions from all models
    resnet_preds = models_dict['resnet'](img)
    efficientnet_preds = models_dict['efficientnetb5'](img)
    vgg_preds = models_dict['vgg16'](img)
    mobilenet_preds = models_dict['mobilenet'](img)

    # Ensemble using averaging
    ensemble_preds = (resnet_preds + efficientnet_preds + mobilenet_preds ) / 3 # Averaging the results

    # Final predictions (sigmoid to get the confidence)
    final_predictions = (ensemble_preds > 0.5).float().item()
    confidence = (ensemble_preds*100 ).float().item() # Confidence as percentage
    print(f"{confidence}%")
    result = 'Cancerous' if final_predictions == 1 else 'Non-Cancerous'
    if result == 'Non-Cancerous':
            confidence= 100-confidence
    if confidence >100:
        confidence = 100
    if confidence < 0:
        confidence = 0
    return result, confidence

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

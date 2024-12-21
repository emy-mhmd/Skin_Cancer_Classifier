# Imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model
def create_model():
    model = models.resnet50(weights=None)  # Create ResNet-50 without pretrained weights
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 128),  # Match with checkpoint
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),  # Match binary classification
        torch.nn.Sigmoid()
    )
    return model

# Instantiate the model
model = create_model()

# Load the checkpoint, skipping mismatched keys
# Load the checkpoint with weights_only=True
checkpoint = torch.load('./models/resnet50.pth', map_location=torch.device('cpu'))

state_dict = checkpoint

# Remove mismatched keys
model_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
model_state_dict.update(filtered_state_dict)

# Load compatible weights
model.load_state_dict(model_state_dict, strict=False)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img)
            pred = (output > 0.5).float().item()
        
        result = 'Cancerous' if pred == 1 else 'Non-Cancerous'
        return jsonify({'prediction': result})
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)

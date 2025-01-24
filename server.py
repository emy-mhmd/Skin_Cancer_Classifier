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

# Define HandcraftedCNN model class

# Handcrafted cnn 

class HandcraftedCNN(torch.nn.Module):
    def __init__(self, device):
        super(HandcraftedCNN, self).__init__()

        # First convolutional block
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolutional block
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third convolutional block
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fourth convolutional block
        '''self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )'''

        # Fully connected layers
        # Flatten size: 256 channels * 14 * 14 = 50176 for 224x224 input
        '''self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 14 * 14, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )'''

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )


        # Move model to device
        self.to(device)

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc(x)

        return x

# Handcrafted cnn 1
'''class HandcraftedCNN(torch.nn.Module):
    def __init__(self, device):
        super(HandcraftedCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(50176, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x'''


# Function to create models
def create_model(model_name, device):
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
        model._fc = torch.nn.Sequential(
            torch.nn.Linear(model._fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.last_channel, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    elif model_name == 'handcraftedcnn':
        model = HandcraftedCNN(device)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_dict = {
    'resnet': load_model_weights(create_model('resnet', device), 'resnet'),
    'efficientnetb5': load_model_weights(create_model('efficientnetb5', device), 'efficientnetb5'),
    'vgg16': load_model_weights(create_model('vgg16', device), 'vgg16'),
    'mobilenet': load_model_weights(create_model('mobilenet', device), 'mobilenet'),
    'handcraftedcnn': create_model('handcraftedcnn', device),  # Add handcrafted CNN model here
}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
            confidence = (output * 100).float().item()  # Confidence as percentage
            print(f"{confidence}%")

        result = 'Cancerous' if pred == 1 else 'Non-Cancerous'
        if result == 'Non-Cancerous':
            confidence = 100 - confidence
        if confidence > 100:
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
    ensemble_preds = (resnet_preds + efficientnet_preds + mobilenet_preds + vgg_preds) / 4  # Averaging the results

    # Final predictions (sigmoid to get the confidence)
    final_predictions = (ensemble_preds > 0.5).float().item()
    confidence = (ensemble_preds * 100).float().item()  # Confidence as percentage
    print(f"{confidence}%")
    result = 'Cancerous' if final_predictions == 1 else 'Non-Cancerous'
    if result == 'Non-Cancerous':
        confidence = 100 - confidence
    if confidence > 100:
        confidence = 100
    if confidence < 0:
        confidence = 0
    return result, confidence

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

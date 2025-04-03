import torch
from torchvision.models import efficientnet_b0
from torch import nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and prepare the model
model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("brain_tumor_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Create a dummy input with the same shape as model's expected input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "brain_tumor_classifier.onnx", 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

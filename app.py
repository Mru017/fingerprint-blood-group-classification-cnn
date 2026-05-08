import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

# ── CLASS NAMES ──────────────────────────────────────────
CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ── MODEL ────────────────────────────────────────────────
class FastCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ── LOAD MODEL ───────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = FastCNN().to(DEVICE)

model.load_state_dict(
    torch.load("fastcnn_model.pth", map_location=DEVICE)
)

model.eval()

# ── TRANSFORM ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ── PREDICTION FUNCTION ──────────────────────────────────
def predict(image):

    image = image.convert("L")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    prediction = CLASS_NAMES[probs.argmax().item()]

    confidence = float(probs.max()) * 100

    return f"Predicted Blood Group: {prediction}\nConfidence: {confidence:.2f}%"

# ── GRADIO UI ────────────────────────────────────────────
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Blood Group Detection from Fingerprint",
    description="Upload a fingerprint image to predict blood group."
)

interface.launch()
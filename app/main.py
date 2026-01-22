from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
from model.model import get_model
from pathlib import Path
from torchvision import transforms

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model().to(device)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "pneumonia.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

classes = ["NORMAL", "PNEUMONIA"]

# Preprocess function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        prob = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(prob, dim=1).item()
        confidence = prob[0][prediction].item()

    return {
        "prediction": classes[prediction],
        "confidence": round(confidence, 4)
    }

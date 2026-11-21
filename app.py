import gradio as gr
import torch
import numpy as np
from PIL import Image
from model import BaselineCODModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = BaselineCODModel().to(device)

checkpoint = torch.load(
    "model_best .pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint, strict=True)
model.eval()


def predict(image):
    if image is None:
        return None

    # Image preprocessing
    image = image.convert("RGB")
    original_size = image.size

    image = image.resize((352, 352))
    image_array = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_array = (image_array - mean) / std

    tensor = torch.from_numpy(image_array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)

    # Prediction
    with torch.no_grad():
        prediction = model(tensor)

    mask = prediction.squeeze().cpu().numpy()

    # Convert prediction into displayable mask
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask = mask.resize(original_size)

    return mask


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Predicted Mask"),
    title="PureDrop Dairy | Camouflaged Object Detection",
    description="Upload an image to generate a camouflaged object detection mask.",
    submit_btn="Detect Object",
    clear_btn="Clear"
)

if __name__ == "__main__":
  demo.launch(share=True)
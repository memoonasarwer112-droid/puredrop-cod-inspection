import torch
from torchvision import transforms
from PIL import Image
from model import BaselineCODModel

def evaluate_model():
    # 1. Device set karna
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Model load karna
    model = BaselineCODModel().to(device)
    model.eval()

    # 3. Test transform define karna
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # 4. Sample image par prediction run karna (jo humne pehle banayi thi)
    try:
        image_path = "data/train/images/sample1.jpg"
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        print("Evaluation completed successfully!")
        print(f"Output prediction shape: {output.shape}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()
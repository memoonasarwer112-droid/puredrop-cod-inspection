import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import py_sod_metrics

from model import BaselineCODModel


# ---------------- SETTINGS ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "data/val/images"
MASK_DIR = "data/val/masks"
CHECKPOINT = "model_best .pth"
DOCS_DIR = "docs"
IMAGE_SIZE = 352

os.makedirs(DOCS_DIR, exist_ok=True)


# ---------------- LOAD MODEL ----------------
model = BaselineCODModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- FIND IMAGE PAIRS ----------------
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = {
    os.path.splitext(f)[0].lower(): f
    for f in os.listdir(IMAGE_DIR)
    if os.path.splitext(f)[1].lower() in extensions
}

masks = {
    os.path.splitext(f)[0].lower(): f
    for f in os.listdir(MASK_DIR)
    if os.path.splitext(f)[1].lower() in extensions
}

common_names = sorted(set(images) & set(masks))

if not common_names:
    raise RuntimeError(
        "No matching image/mask filenames found in validation folders."
    )

print(f"Found {len(common_names)} validation image pairs.")


# ---------------- METRICS ----------------
sm = py_sod_metrics.Smeasure()
em = py_sod_metrics.Emeasure()
wfm = py_sod_metrics.WeightedFmeasure()
mae = py_sod_metrics.MAE()

visualizations = []


# ---------------- EVALUATION ----------------
for index, name in enumerate(common_names):

    image_path = os.path.join(IMAGE_DIR, images[name])
    mask_path = os.path.join(MASK_DIR, masks[name])

    original_image = Image.open(image_path).convert("RGB")
    ground_truth = Image.open(mask_path).convert("L")

    original_size = ground_truth.size

    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = (prediction * 255).clip(0, 255).astype(np.uint8)

    prediction_image = Image.fromarray(prediction)
    prediction_image = prediction_image.resize(
        original_size, Image.Resampling.BILINEAR
    )

    prediction_array = np.array(prediction_image)
    gt_array = np.array(ground_truth)

    # Binary ground truth
    gt_array = (gt_array > 127).astype(np.uint8) * 255

    # Calculate metrics
    sm.step(pred=prediction_array, gt=gt_array)
    em.step(pred=prediction_array, gt=gt_array)
    wfm.step(pred=prediction_array, gt=gt_array)
    mae.step(pred=prediction_array, gt=gt_array)

    # Save first 3 visual comparisons
    if index < 3:
        display_image = original_image.resize(original_size)

        canvas = Image.new(
            "RGB",
            (original_size[0] * 3, original_size[1] + 35),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        canvas.paste(display_image, (0, 35))
        canvas.paste(
            Image.fromarray(gt_array).convert("RGB"),
            (original_size[0], 35)
        )
        canvas.paste(
            prediction_image.convert("RGB"),
            (original_size[0] * 2, 35)
        )

        draw.text((10, 10), "Input Image", fill="black")
        draw.text((original_size[0] + 10, 10), "Ground Truth", fill="black")
        draw.text((original_size[0] * 2 + 10, 10), "Prediction", fill="black")

        output_path = os.path.join(
            DOCS_DIR, f"prediction_{index + 1}.png"
        )
        canvas.save(output_path)
        visualizations.append(output_path)


# ---------------- COLLECT RESULTS ----------------
sm_result = sm.get_results()
em_result = em.get_results()
wfm_result = wfm.get_results()
mae_result = mae.get_results()

results = {
    "S-measure": float(sm_result["sm"]),
    "E-measure (adaptive)": float(em_result["em"]["adp"]),
    "Weighted F-measure": float(wfm_result["wfm"]),
    "MAE": float(mae_result["mae"])
}


# ---------------- CREATE REPORT ----------------
table = (
    "| Metric | Validation Result |\n"
    "|---|---:|\n"
)

for metric, value in results.items():
    table += f"| {metric} | {value:.6f} |\n"

report = (
    "# Model Evaluation Results\n\n"
    f"**Validation image pairs:** {len(common_names)}\n\n"
    + table
    + "\n## Prediction Visualizations\n\n"
)

for path in visualizations:
    report += f"![{os.path.basename(path)}]({path})\n\n"

with open("evaluation_results.md", "w", encoding="utf-8") as file:
    file.write(report)

# Update README without duplicating the section
readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as file:
        readme = file.read()

    marker = "<!-- EVALUATION RESULTS -->"
    section = marker + "\n" + report

    if marker in readme:
        readme = readme.split(marker)[0].rstrip()

    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(readme.rstrip() + "\n\n" + section)

print("\n========== EVALUATION RESULTS ==========")
for metric, value in results.items():
    print(f"{metric}: {value:.6f}")

print("\nReport saved: evaluation_results.md")
print("Visualizations saved in:", DOCS_DIR)
print("README updated (if present).")
from pathlib import Path 
import torch 
import numpy as np
import csv
import cv2
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image 
from transformers import AutoImageProcessor, SiglipForImageClassification 

MODEL_PATH = Path("/Users/ericliu/CAC Stuff/DeepfakeModel_2.1_60KHFDataset") 
IMAGE_FOLDER = Path("/Users/ericliu/CAC Stuff/images to test") 
RESULT_FOLDER = Path("/Users/ericliu/CAC Stuff/DeepfakeModelResults") 

if torch.backends.mps.is_available(): 
    device = torch.device("mps") 
elif torch.cuda.is_available(): 
    device = torch.device("cuda") 
else: 
    device = torch.device("cpu") 

extensions = { ".jpg", ".jpeg", ".png", } 

CAM_TARGET = 0
EIGEN_SMOOTH = False

processor = AutoImageProcessor.from_pretrained(MODEL_PATH) 

model = SiglipForImageClassification.from_pretrained( MODEL_PATH ) 
model.to(device) 
model.eval() 

class SiglipCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(
            pixel_values=pixel_values
        )

        return outputs.logits

cam_model = SiglipCAMWrapper(model).to(device)
cam_model.eval()

target_layers = [
    model.vision_model.encoder.layers[-1].layer_norm1
]

def reshape_transform(tensor):
    batch_size, num_tokens, channels = tensor.shape

    height = int(np.sqrt(num_tokens))
    width = height

    if height * width != num_tokens:
        raise ValueError(
            f"Cannot reshape {num_tokens} tokens into a square grid."
        )

    result = tensor.reshape(
        batch_size,
        height,
        width,
        channels
    )

    result = result.permute(
        0,
        3,
        1,
        2
    )

    return result

image_paths = sorted( p for p in IMAGE_FOLDER.rglob("*") if p.is_file() and p.suffix.lower() in extensions ) 
print(f"Scanning {len(image_paths)} images.") 

results = [] 
cam = GradCAM(
    model=cam_model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)

for i, image_path in enumerate(image_paths, start=1): 
    try: 
        image = Image.open(image_path).convert("RGB") 

        original_width, original_height = image.size

        inputs = processor( images=image, return_tensors="pt" ) 
        inputs = { key: value.to(device) for key, value in inputs.items() } 

        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad(): 
            outputs = model(pixel_values=pixel_values) 
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1) 

        predicted_class = probabilities.argmax(dim=-1).item() 
        confidence = probabilities[0, predicted_class].item() 
        fake_probability = probabilities[0, 0].item() 
        real_probability = probabilities[0, 1].item() 
        label = model.config.id2label[predicted_class] 

        if CAM_TARGET == "predicted":
            cam_class = predicted_class
        elif CAM_TARGET in (0,1):
            cam_class = CAM_TARGET
        else:
            raise ValueError("CAM_TARGET must be 'predicted', 0 or 1")

        cam_label = model.config.id2label[cam_class]

        targets = [ClassifierOutputTarget(cam_class)]

        grayscale_cam = cam(
            input_tensor=pixel_values,
            targets=targets,
            eigen_smooth=EIGEN_SMOOTH,
            aug_smooth=False,
        )

        grayscale_cam = grayscale_cam[0]

        display_size = (
            pixel_values.shape[-1],
            pixel_values.shape[-2]
        )

        display_image = image.resize(
            display_size,
            Image.Resampling.BILINEAR
        )

        rgb_array = np.asarray(
            display_image
        ).astype(np.float32) / 255.0

        cam_uint8 = (
            np.clip(
                grayscale_cam,
                0.0,
                1.0
            ) * 255
        ).astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(
            cam_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

        rgb_uint8 = (
            rgb_array * 255
        ).astype(np.uint8)

        overlay = cv2.addWeighted(
            rgb_uint8,
            0.55,
            heatmap_rgb,
            0.45,
            0
        )

        heatmap_original = cv2.resize(
            heatmap_rgb,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        overlay_original = cv2.resize(
            overlay,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        relative_path = image_path.relative_to(
            IMAGE_FOLDER
        )

        output_dir = (
            RESULT_FOLDER
            / relative_path.parent
            / relative_path.stem
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        original_output_path = (
            output_dir
            / f"original{image_path.suffix.lower()}"
        )

        image.save(
            original_output_path
        )

        heatmap_output_path = (
            output_dir
            / "heatmap.png"
        )

        Image.fromarray(
            heatmap_original
        ).save(
            heatmap_output_path
        )

        overlay_output_path = (
            output_dir
            / "overlay.png"
        )

        Image.fromarray(
            overlay_original
        ).save(
            overlay_output_path
        )


        result_data = {
            "file": str(image_path),
            "prediction": label,
            "prediction_class": predicted_class,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,

            "gradcam_target": cam_label,
            "gradcam_target_class": cam_class,

            "original_width": original_width,
            "original_height": original_height,

            "model_path": str(MODEL_PATH),
            "device": str(device),

            "target_layer":
                "vision_model.encoder.layers[-1].layer_norm1",

            "cam_method": "GradCAM",
        }

        result_json_path = (
            output_dir
            / "result.json"
        )

        with open(
            result_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result_data,
                f,
                indent=4
            )

        results.append({
            "file": str(image_path),
            "prediction": label,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "gradcam_target": cam_label,
            "result_folder": str(output_dir),
        })
        
        print(
            f"[{i}/{len(image_paths)}] "
            f"{image_path.name} = {label} "
            f"(Fake: {fake_probability:.2%}, "
            f"Real: {real_probability:.2%}) "
            f"| CAM: {cam_label}"
        )


    except Exception as e:

        print(
            f"[{i}/{len(image_paths)}] "
            f"ERROR: {image_path}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

summary_csv = (
    RESULT_FOLDER
    / "summary.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "prediction",
            "confidence",
            "fake_probability",
            "real_probability",
            "gradcam_target",
            "result_folder",
        ]
    )

    writer.writeheader()

    writer.writerows(results)

print()
print("Finished.")
print(f"Results saved to: {RESULT_FOLDER}")
print(f"Summary saved to: {summary_csv}")

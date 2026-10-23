import argparse
import math
from pathlib import Path
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "Deepfake_Model"
if not DEFAULT_MODEL_DIR.is_dir():
    DEFAULT_MODEL_DIR = SCRIPT_DIR

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}


class DeepfakeDetector(nn.Module):
    def __init__(self, clip_vision, fft_embed_dim=128, num_classes=2):
        super().__init__()
        self.clip_vision = clip_vision
        clip_hidden = clip_vision.config.hidden_size

        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, fft_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(clip_hidden + fft_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.id2label = {0: "Realism", 1: "Deepfake"}

    def forward(self, pixel_values):
        clip_embed = self.clip_vision(pixel_values=pixel_values).pooler_output
        gray = pixel_values.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(torch.fft.fftshift(fft)))
        fft_embed = self.fft_branch(fft_mag)
        fused = torch.cat([clip_embed, fft_embed], dim=1)
        return self.classifier(fused)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir, device):
    model_dir = Path(model_dir).expanduser().resolve()
    clip_config = model_dir / "config.json"
    head_path = model_dir / "head_weights.pt"
    processor_config = model_dir / "preprocessor_config.json"

    clip = CLIPVisionModel.from_pretrained(str(model_dir), local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)

    model = DeepfakeDetector(clip)
    model.fft_branch.load_state_dict(checkpoint["fft_branch"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    if "id2label" in checkpoint:
        model.id2label = {int(k): str(v) for k, v in checkpoint["id2label"].items()}

    model.to(device).eval()
    return model, processor


def predict_batch_with_gradcam(images, model, processor, device):
    vision_backbone = getattr(model.clip_vision, "vision_model", model.clip_vision)
    layers = vision_backbone.encoder.layers
    target_layer = layers[-2] if len(layers) > 1 else layers[-1]
    captured = {}

    def save_activation(_module, _inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        activation.retain_grad()
        captured["activation"] = activation

    hook = target_layer.register_forward_hook(save_activation)
    try:
        inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        probabilities = torch.softmax(logits.detach(), dim=1).cpu()
        target_classes = logits.detach().argmax(dim=1)
        logits.gather(1, target_classes.unsqueeze(1)).sum().backward()

        activations = captured["activation"]
        gradients = activations.grad
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]
        weights = gradients.mean(dim=1, keepdim=True)
        cams = torch.relu((weights * activations).sum(dim=-1))

        grid_size = math.isqrt(cams.shape[1])
        cams = cams.reshape(cams.shape[0], 1, grid_size, grid_size)
        cams = F.interpolate(
            cams,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cams = cams - cams.amin(dim=(1, 2), keepdim=True)
        cams = cams / cams.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return probabilities, [cam.detach().cpu().numpy() for cam in cams]
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def make_overlay(original_bgr, cam):
    height, width = original_bgr.shape[:2]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)


def video_source_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def sampled_video_frames(video_path, sample_fps):
    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / sample_fps
    next_sample_time = 0.0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / source_fps
            if timestamp + 1e-9 >= next_sample_time:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb), frame.copy()
                next_sample_time += interval
                while next_sample_time <= timestamp:
                    next_sample_time += interval

            frame_index += 1
    finally:
        cap.release()


def report(prob_sum, count, model, is_video):

    mean_probs = prob_sum / count
    predicted_id = int(mean_probs.argmax().item())
    labels = model.id2label
    label = labels.get(predicted_id, str(predicted_id))
    confidence = mean_probs[predicted_id].item() * 100

    if is_video:
        print(f"Frames checked: {count}")
    print(f"Prediction: {label} ({confidence:.1f}%)")
    print("Average class probabilities:")
    for class_id, probability in enumerate(mean_probs.tolist()):
        print(f"  {labels.get(class_id, str(class_id))}: {probability * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run the local deepfake detector on an image or video and save Grad-CAM overlays"
    )
    parser.add_argument("input", help="image/video path")
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR),
        help="Folder containing config.json, model.safetensors, preprocessor_config.json, and head_weights.pt",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0,
        help="frames the model checks per second (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="number of frames to process simultaneously (default: 8)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = model_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (f"{input_path.stem}_gradcam.mp4" if is_video else f"{input_path.stem}_gradcam.png")

    device = choose_device()
    print(f"Device: {device}")
    model, processor = load_model(args.model_dir, device)

    if is_video:
        source_fps = video_source_fps(input_path)
        output_fps = min(args.fps, source_fps)
        images = sampled_video_frames(input_path, args.fps)
        print(f"Output video: {output_path} ({output_fps:g} FPS)")
    else:
        with Image.open(input_path) as image_file:
            image = image_file.convert("RGB")
        original_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        images = [(image, original_bgr)]
        output_fps = None
        print(f"Heatmap image: {output_path}")

    prob_sum = None
    count = 0
    batch = []
    video_writer = None
    data = []
    if(is_video):
        data.append("Analyzing Video\n")
    else:
        data.append("Analyzing Image\n")

    def process_batch(items):
        nonlocal prob_sum, count, video_writer

        pil_images = [item[0] for item in items]
        probs, cams = predict_batch_with_gradcam(
            pil_images, model, processor, device
        )

        prob_sum = probs.sum(0) if prob_sum is None else prob_sum + probs.sum(0)

        labels = model.id2label
        print("\nIndividual frame predictions:")

        for i, (item, frame_probs, cam) in enumerate(zip(items, probs, cams)):
            original_image, original_bgr = item

            frame_number = count + i + 1
            predicted_id = int(frame_probs.argmax().item())
            predicted_label = labels.get(predicted_id, str(predicted_id))
            confidence = frame_probs[predicted_id].item() * 100

            if is_video:
                # Estimate timestamp from the sampled frame sequence.
                timestamp = (count + i) / args.fps
                print(
                    f"Frame {frame_number:5d} | "
                    f"Time: {timestamp:8.3f}s | "
                    f"Prediction: {predicted_label} ({confidence:.1f}%)"
                )
                data.append(f"Frame Number:{frame_number:5d} at time {timestamp:8.3f} seconds ")
                data.append("\n")
                data.append(f"Prediction: {predicted_label} ({confidence:.1f}%)")
                data.append("\n")
            else:
                print(
                    f"Prediction: {predicted_label} ({confidence:.1f}%)"
                )
                data.append(f"Prediction: {predicted_label} ({confidence:.1f}%)")
                data.append("\n")

            for class_id, probability in enumerate(frame_probs.tolist()):
                print(
                    f"      {labels.get(class_id, str(class_id))}: "
                    f"{probability * 100:.2f}%"
                )

            overlay = make_overlay(original_bgr, cam)

            if is_video:
                if video_writer is None:
                    height, width = overlay.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        output_fps,
                        (width, height),
                    )
                video_writer.write(overlay)

        count += len(items)


    try:
        for item in images:
            batch.append(item)
            if len(batch) >= args.batch_size:
                process_batch(batch)
                batch.clear()
        if batch:
            process_batch(batch)
    finally:
        if video_writer is not None:
            video_writer.release()

    report(prob_sum, count, model, is_video)
    print(f"Grad-CAM output saved to: {output_path}")
    file_path = output_dir / f"{input_path.stem}_Data.json"
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()

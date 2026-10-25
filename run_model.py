import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

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
    head_path = model_dir / "head_weights.pt"

    clip = CLIPVisionModel.from_pretrained(str(model_dir), local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)

    model = DeepfakeDetector(clip)
    model.fft_branch.load_state_dict(checkpoint["fft_branch"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    if "id2label" in checkpoint:
        model.id2label = {int(k): str(v) for k, v in checkpoint["id2label"].items()}

    model.to(device).eval()

    # This is an inference-only script. Frozen parameters avoid storing parameter-gradient
    # state; Grad-CAM creates gradients only for the activation it actually needs.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, processor


def _replace_first_output(output, activation):
    """Return a module output with its first tensor replaced by the CAM leaf tensor."""
    if isinstance(output, tuple):
        return (activation, *output[1:])
    if isinstance(output, list):
        return [activation, *output[1:]]
    return activation


def predict_batch(images, model, processor, device, compute_gradcam=True):
    # CPU preprocessing is performed once for the entire batch.
    inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(
        device, non_blocking=(device.type == "cuda")
    )

    if not compute_gradcam:
        # Fast inference path: no autograd graph is built at all.
        with torch.inference_mode():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu()
        return probabilities, None

    vision_backbone = getattr(model.clip_vision, "vision_model", model.clip_vision)
    layers = vision_backbone.encoder.layers
    target_layer = layers[-2] if len(layers) > 1 else layers[-1]
    captured = {}

    def save_activation(_module, _inputs, output):
        # Detach at the selected layer and restart autograd from this activation.
        # Since model weights are frozen, earlier CLIP layers run without building a
        # backward graph. Gradients are then calculated only through the layers after
        # this target, which is sufficient for Grad-CAM.
        value = output[0] if isinstance(output, (tuple, list)) else output
        activation = value.detach().requires_grad_(True)
        captured["activation"] = activation
        return _replace_first_output(output, activation)

    hook = target_layer.register_forward_hook(save_activation)
    try:
        # Do not use inference_mode here: Grad-CAM needs gradients for one activation.
        with torch.enable_grad():
            logits = model(inputs)
            probabilities = torch.softmax(logits.detach(), dim=1).cpu()
            target_classes = logits.detach().argmax(dim=1)
            target_scores = logits.gather(1, target_classes.unsqueeze(1)).sum()
            gradients, = torch.autograd.grad(
                target_scores,
                captured["activation"],
                retain_graph=False,
                create_graph=False,
            )

        # CAM post-processing itself needs no gradients.
        with torch.no_grad():
            activations = captured["activation"].detach()[:, 1:, :]
            gradients = gradients.detach()[:, 1:, :]
            weights = gradients.mean(dim=1, keepdim=True)
            cams = torch.relu((weights * activations).sum(dim=-1))

            grid_size = math.isqrt(cams.shape[1])
            if grid_size * grid_size != cams.shape[1]:
                raise RuntimeError(
                    f"Grad-CAM expected a square patch grid, got {cams.shape[1]} patches."
                )
            cams = cams.reshape(cams.shape[0], 1, grid_size, grid_size)
            cams = F.interpolate(
                cams,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            cams = cams - cams.amin(dim=(1, 2), keepdim=True)
            cams = cams / cams.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return probabilities, [cam.cpu().numpy() for cam in cams]
    finally:
        hook.remove()


def make_overlay(original_bgr, cam):
    height, width = original_bgr.shape[:2]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)


def video_source_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Could not determine a valid source FPS for video: {video_path}")
    return fps


def sampled_video_frames(video_path, sample_fps):
    """Yield sampled frames; retrieve pixel data only for frames that will be analyzed."""
    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise ValueError(f"Could not determine a valid source FPS for video: {video_path}")

    interval = 1.0 / sample_fps
    next_sample_time = 0.0
    frame_index = 0

    try:
        while cap.grab():
            timestamp = frame_index / source_fps
            if timestamp + 1e-9 >= next_sample_time:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(rgb), frame.copy()
                next_sample_time += interval
                while next_sample_time <= timestamp:
                    next_sample_time += interval
            frame_index += 1
    finally:
        cap.release()


def find_deepfake_class_id(model):
    """Find the class index corresponding to deepfake/fake in the model labels."""
    labels = model.id2label
    for class_id, label in labels.items():
        normalized = str(label).strip().lower()
        if "deepfake" in normalized or ("fake" in normalized and "real" not in normalized):
            return int(class_id)

    if len(labels) > 1 and 1 in labels:
        return 1
    raise ValueError(
        "Could not determine the Deepfake class from model.id2label. "
        "Please ensure the checkpoint labels identify the deepfake class."
    )


def summarize_probabilities(probabilities, labels):
    """Convert a class-probability tensor into a JSON-serializable summary."""
    predicted_id = int(probabilities.argmax().item())
    return {
        "prediction": labels.get(predicted_id, str(predicted_id)),
        "confidence_percent": round(float(probabilities[predicted_id].item()) * 100, 4),
        "class_probabilities_percent": {
            labels.get(class_id, str(class_id)): round(float(probability) * 100, 4)
            for class_id, probability in enumerate(probabilities.tolist())
        },
    }


def report(frame_probabilities, count, model, is_video, top_k_percent):
    """Compute and print both all-frame-average and top-k video results."""
    if not frame_probabilities:
        raise ValueError("No frame probabilities were collected; cannot report a prediction.")

    all_probs = torch.stack(frame_probabilities)
    labels = model.id2label
    average_probs = all_probs.mean(dim=0)
    average_result = summarize_probabilities(average_probs, labels)

    deepfake_class_id = find_deepfake_class_id(model)
    if is_video:
        k = max(1, math.ceil(count * top_k_percent / 100.0))
        k = min(k, count)
        top_indices = torch.topk(
            all_probs[:, deepfake_class_id], k=k, largest=True
        ).indices
        top_k_probs = all_probs[top_indices].mean(dim=0)
        print(f"Frames checked: {count}")
        print(
            f"Frames used for top-k result: {k} ({top_k_percent:g}% top-scoring frames, "
            f"ranked by {labels.get(deepfake_class_id, 'Deepfake')} probability)"
        )
    else:
        k = count
        top_k_probs = average_probs

    top_k_result = summarize_probabilities(top_k_probs, labels)

    print("\nAverage probability result (all sampled frames):")
    print(f"Prediction: {average_result['prediction']} ({average_result['confidence_percent']:.1f}%)")
    for label, probability in average_result["class_probabilities_percent"].items():
        print(f"  {label}: {probability:.2f}%")

    print("\nTop-k sampling result:")
    print(f"Prediction: {top_k_result['prediction']} ({top_k_result['confidence_percent']:.1f}%)")
    for label, probability in top_k_result["class_probabilities_percent"].items():
        print(f"  {label}: {probability:.2f}%")

    return {
        "frames_checked": int(count),
        "average_probability_result": average_result,
        "top_k_sampling_result": {
            **top_k_result,
            "top_k_percent": float(top_k_percent) if is_video else None,
            "frames_used": int(k),
            "selection_rule": (
                "Average the top-scoring frames ranked by Deepfake probability"
                if is_video else
                "Single-image result; identical to the average probability result"
            ),
        },
    }


def append_result_to_json(results_path, run_record):
    """Append one run summary while preserving all existing JSON content."""
    results_path = Path(results_path)
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    else:
        existing = []

    if isinstance(existing, list):
        existing.append(run_record)
        payload = existing
    elif isinstance(existing, dict):
        if isinstance(existing.get("runs"), list):
            existing["runs"].append(run_record)
        elif isinstance(existing.get("results"), list):
            existing["results"].append(run_record)
        else:
            existing["runs"] = [run_record]
        payload = existing
    else:
        raise ValueError(
            f"Cannot append to {results_path}: expected a JSON list or object at the root."
        )

    temporary_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary_path, results_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the local deepfake detector on an image or video and optionally save Grad-CAM overlays"
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
        help="number of frames to process simultaneously (default: 4)",
    )
    parser.add_argument(
        "--top-k-percent", type=float, default=30.0,
        help="for videos, average only the top-scoring percentage of frames by Deepfake probability (default: 30)",
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="skip Grad-CAM and overlay writing for faster prediction-only inference",
    )
    parser.add_argument(
        "--quiet-frames", action="store_true",
        help="suppress per-frame console output (per-frame data is still saved)",
    )
    args = parser.parse_args()

    if not 0 < args.top_k_percent <= 100:
        parser.error("--top-k-percent must be greater than 0 and at most 100")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist or is not a file: {input_path}")

    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = model_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{input_path.stem}_gradcam.mp4" if is_video else f"{input_path.stem}_gradcam.png"
    )

    device = choose_device()
    print(f"Device: {device}")
    model, processor = load_model(args.model_dir, device)

    if is_video:
        source_fps = video_source_fps(input_path)
        output_fps = min(args.fps, source_fps)
        images = sampled_video_frames(input_path, args.fps)
        if args.no_gradcam:
            print("Grad-CAM disabled; no overlay video will be written.")
        else:
            print(f"Output video: {output_path} ({output_fps:g} FPS)")
    else:
        with Image.open(input_path) as image_file:
            image = image_file.convert("RGB")
        original_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        images = [(image, original_bgr)]
        output_fps = None
        if args.no_gradcam:
            print("Grad-CAM disabled; no overlay image will be written.")
        else:
            print(f"Heatmap image: {output_path}")

    all_probabilities = []
    count = 0
    batch = []
    video_writer = None
    data = ["Analyzing Video\n" if is_video else "Analyzing Image\n"]

    def process_batch(items):
        nonlocal count, video_writer

        pil_images = [item[0] for item in items]
        probs, cams = predict_batch(
            pil_images, model, processor, device, compute_gradcam=not args.no_gradcam
        )
        all_probabilities.extend(frame_probs for frame_probs in probs)
        labels = model.id2label

        if not args.quiet_frames:
            print("\nIndividual frame predictions:")

        for i, (item, frame_probs) in enumerate(zip(items, probs)):
            _, original_bgr = item
            frame_number = count + i + 1
            frame_values = frame_probs.tolist()
            predicted_id = int(frame_probs.argmax().item())
            predicted_label = labels.get(predicted_id, str(predicted_id))
            confidence = frame_values[predicted_id] * 100

            if is_video:
                timestamp = (count + i) / args.fps
                if not args.quiet_frames:
                    print(
                        f"Frame {frame_number:5d} | Time: {timestamp:8.3f}s | "
                        f"Prediction: {predicted_label} ({confidence:.1f}%)"
                    )
                data.extend([
                    f"Frame Number:{frame_number:5d} at time {timestamp:8.3f} seconds ",
                    "\n",
                    f"Prediction: {predicted_label} ({confidence:.1f}%)",
                    "\n",
                ])
            else:
                if not args.quiet_frames:
                    print(f"Prediction: {predicted_label} ({confidence:.1f}%)")
                data.extend([f"Prediction: {predicted_label} ({confidence:.1f}%)", "\n"])

            if not args.quiet_frames:
                for class_id, probability in enumerate(frame_values):
                    print(f"      {labels.get(class_id, str(class_id))}: {probability * 100:.2f}%")

            if cams is not None:
                overlay = make_overlay(original_bgr, cams[i])
                if is_video:
                    if video_writer is None:
                        height, width = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(output_path), fourcc, output_fps, (width, height)
                        )
                        if not video_writer.isOpened():
                            video_writer.release()
                            video_writer = None
                            raise RuntimeError(f"Could not create output video: {output_path}")
                    video_writer.write(overlay)
                elif not is_video:
                    if not cv2.imwrite(str(output_path), overlay):
                        raise RuntimeError(f"Could not write Grad-CAM image: {output_path}")

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

    summary = report(all_probabilities, count, model, is_video, args.top_k_percent)
    if not args.no_gradcam:
        print(f"Grad-CAM output saved to: {output_path}")

    data_path = output_dir / f"{input_path.stem}_Data.json"
    with data_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    results_json_path = output_dir / f"{input_path.stem}_Final_Results.json"
    run_record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "input_type": "video" if is_video else "image",
        "sampling_fps": float(args.fps) if is_video else None,
        "top_k_percent_requested": float(args.top_k_percent) if is_video else None,
        **summary,
    }
    append_result_to_json(results_json_path, run_record)
    print(f"Aggregate results appended to: {results_json_path}")


if __name__ == "__main__":
    main()

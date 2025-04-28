#!/usr/bin/env python3
import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(path, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def main():
    parser = argparse.ArgumentParser(
        description="Interpolation / Morphing experiment between hero and villain prototypes")
    parser.add_argument("--model-path",   required=True,
                        help="Path to your single-logit .pth checkpoint")
    parser.add_argument("--hero-img",     required=True,
                        help="Path to the hero prototype PNG")
    parser.add_argument("--villain-img",  required=True,
                        help="Path to the villain prototype PNG")
    parser.add_argument("--steps",   type=int,   default=21,
                        help="Number of interpolation points (default: 21)")
    parser.add_argument("--target",  choices=["hero","villain"], required=True,
                        help="Which class’s probability to plot")
    parser.add_argument("--output",  required=True,
                        help="Path to save the resulting plot (PNG)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load model
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # 2) Load images
    hero    = load_image(args.hero_img,    device)  # [1,3,224,224]
    villain = load_image(args.villain_img, device)

    # 3) Interpolate and collect probabilities
    alphas = np.linspace(0.0, 1.0, args.steps)
    probs  = []
    delta = 1.0 if args.target == "villain" else -1.0

    with torch.no_grad():
        for a in alphas:
            x = (1 - a) * hero + a * villain
            logit = model(x).squeeze()               # scalar
            prob  = torch.sigmoid(delta * logit).item()
            probs.append(prob)

    # 4) Plot
    plt.figure(figsize=(6,4))
    plt.plot(alphas, probs, marker='o')
    plt.xlabel(r'Interpolation factor $\alpha$ (0 = hero, 1 = villain)')
    plt.ylabel(f"P({args.target})")
    plt.title(f"Hero→Villain Interpolation ({args.target} probability)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved interpolation plot to {args.output}")

if __name__ == "__main__":
    main()

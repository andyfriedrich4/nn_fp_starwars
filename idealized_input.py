#!/usr/bin/env python3
import os
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image

def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def synthesize(model, is_villain, device,
               lr=0.1, iters=200,
               tv_w=1e-4, l2_w=1e-6,
               init_x=None, output_prefix=None):
    """
    Performs gradient ascent to synthesize a hero/villain prototype.
    Saves an image every 50 iterations if output_prefix is provided.
    """
    # Initialize x
    if init_x is not None:
        x = init_x.clone().detach().requires_grad_(True)
    else:
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)

    optimizer = optim.Adam([x], lr=lr)
    to_pil = T.ToPILImage()

    for i in range(iters):
        optimizer.zero_grad()
        logit = model(x).squeeze()                   # single scalar
        # compute probability of the target class
        #  - for villain:   P(villain) = σ(logit)
        #  - for hero:      P(hero)    = σ(−logit)
        delta =  1.0 if is_villain else -1.0
        prob  = torch.sigmoid(delta * logit)
        score =  delta * logit
        loss = -score + l2_w * torch.norm(x) + tv_w * total_variation(x)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(-1, 1)

        if i % (iters / 4) == 0:
            print(f"iter {i:03d} logit={logit.item():.3f} loss={loss.item():.3f}")
            if output_prefix:
                snapshot = x.detach().cpu().squeeze(0).clamp(0,1)
                img = to_pil(snapshot)
                img.save(f"{output_prefix}_iter{i}.png")

    return x.detach()

def main():
    parser = argparse.ArgumentParser(
        description="Generate an ‘idealized’ Star Wars hero/villain prototype.")
    parser.add_argument("--model-path",    required=True,
                        help="Path to your .pth checkpoint")
    parser.add_argument("--target",        required=True,
                        choices=["hero","villain"],
                        help="Which prototype to generate")
    parser.add_argument("--init-image",    default=None,
                        help="Optional: path to an image to start from")
    parser.add_argument("--output",        required=True,
                        help="Where to save the final prototype PNG")
    parser.add_argument("--iters",   type=int, default=200,
                        help="Number of optimization iterations")
    parser.add_argument("--lr",      type=float, default=0.1,
                        help="Learning rate for ascent")
    parser.add_argument("--tv-weight", type=float, default=1e-4,
                        help="Total variation regularization weight")
    parser.add_argument("--l2-weight",  type=float, default=1e-6,
                        help="L2 pixel regularization weight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build single-logit ResNet-34
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    is_villain = (args.target == "villain")
    init_x = None

    # Optional warm-start from a real image
    if args.init_image:
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225])
        ])
        img = Image.open(args.init_image).convert("RGB")
        init_x = preprocess(img).unsqueeze(0).to(device)
        print(f"Initializing from image {args.init_image}")

    # Derive prefix for intermediate snapshots
    output_prefix = os.path.splitext(args.output)[0]

    # Run synthesis
    proto = synthesize(model,
                       is_villain,
                       device,
                       lr=args.lr,
                       iters=args.iters,
                       tv_w=args.tv_weight,
                       l2_w=args.l2_weight,
                       init_x=init_x,
                       output_prefix=output_prefix)

    # Convert and save final prototype
    to_pil = T.ToPILImage()
    final_img = to_pil(proto.squeeze(0).cpu().clamp(0,1))
    final_img.save(args.output)
    print(f"Saved {args.target} prototype to {args.output}")

if __name__ == "__main__":
    main()

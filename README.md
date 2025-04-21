Data is contained in the `/data` folder
- **train**: contains all training images
- **test**: contains all validation images (model evaluated on this set after every training loop)
- **unseen**: contains all images used for final evaluation (never touched during training)

`/models` contains all the saved models

Run `python main.py -h` for all run options.
- Example of a training run:`python main.py --batch_size 64 --num_workers 8` 
- Example of an evaluation run: `python main.py --mode eval --load_model models/BEST_resnet34_2025-04-21_13-27-44.pth --batch_size 64 --num_workers 8`
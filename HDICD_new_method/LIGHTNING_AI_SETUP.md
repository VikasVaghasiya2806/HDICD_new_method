# Running HDICD on Lightning AI

This guide explains how to run the new **HDICD** method on [Lightning AI](https://lightning.ai/) Studios. Since you have already pushed this code to GitHub, it is very straightforward to set up.

## Step 1 — Start a GPU Studio
1. Go to [lightning.ai](https://lightning.ai) and sign in.
2. Create or open a **Studio**. For training with the DINO backbone and multiple losses, select the **H100 80GB** GPU instance for maximum performance.
3. Once the Studio loads, open the **Terminal** at the bottom of the screen.

## Step 2 — Clone Your Repository
Since your code is on GitHub, clone it directly into the Studio's workspace:
```bash
cd /teamspace/studios/this_studio
git clone https://github.com/VikasVaghasiya2806/HDICD_new_method.git
cd HDICD_new_method
```

## Step 3 — Install Dependencies
Lightning AI Studios come with PyTorch pre-installed, but you need to make sure the other dependencies are present:
```bash
pip install pyyaml scipy tqdm torchvision numpy
```
*(Note: If you encounter issues with `geoopt` or `hyptorch` from previous setups, this new method includes standard Poincaré math built-in, so external hyperbolic libraries are not strictly required unless you extend the `poincare_ops.py` module.)*

## Step 4 — Prepare Datasets
The training script defaults to `CIFAR100`, which torchvision will auto-download for you into the `./data` folder on the first run.
If you plan to use CUB-200, PACS, or Office-Home, create a `datasets` folder and upload them:
```bash
mkdir -p /teamspace/studios/this_studio/datasets
# Now upload your PACS/CUB zip files through the Lightning AI file browser
# and unzip them into this folder.
```
Then update your `configs/config.yaml` to point to the correct `data_path` for your dataset.

## Step 5 — Run Training
You are now ready to train the model. Simply run:
```bash
python scripts/train.py --config configs/config.yaml
```

### Tips for Lightning AI
- **Background Jobs**: If your training takes a long time, you might want to run it in the background or use `tmux`/`nohup` so it doesn't stop if you close the browser tab.
- **Monitoring**: You can write a quick script to plot the losses printed in the terminal, or modify `scripts/train.py` to import `wandb` (Weights & Biases) which is natively supported on Lightning AI.

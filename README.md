## Prerequisites

The code has been tested with the following environment:

- **Python**: 3.18.6
- **g++**: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
- **PyTorch**: 1.8.1+cu111

Ensure these tools are available in your environment before proceeding.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iballester/spike
   cd spike
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. Compile the CUDA layers required for [PointNet++](http://arxiv.org/abs/1706.02413):
   ```bash
   cd modules
   python setup.py install
   ```

---

## Instructions for ITOP Dataset

1. Download the dataset ITOP SIDE (point clouds and labels) from [ITOP Dataset | Zenodo](https://zenodo.org/record/3932973#.Yp8SIxpBxPA) and unzip the contents.

2. Isolate points corresponding to the human body in the point clouds and save the results as `.npz` files. 
- You can use the provided script `utils/preprocess_itop.py` as an example. This script takes the original `.h5` files, removes the background by clustering and depth thresholding (see the paper for more details) and saves the results as point cloud sequences in `.npz` format. To run this script, make sure you have the open3d library installed.
  
3. Update the `ITOP_SIDE_PATH` variable in `const/path` to point to your dataset location. Structure your dataset directory as follows:

   ```
   dataset_directory/
   ├── test/           # Folder containing .npz files for testing
   ├── train/          # Folder containing .npz files for training
   ├── test_labels.h5  # Labels for the test set
   ├── train_labels.h5 # Labels for the training set
   ```

---

## Usage

### Training

To train the model, check that the config.yaml has the correct parameters and run:

```bash
python train_itop.py --config experiments/ITOP-SIDE/1/config.yaml
```

### Inference

For predictions, update the path pointing to the model weights, check that the config.yaml has the correct parameters and run:

```bash
python predict_itop.py --config experiments/ITOP-SIDE/1/config.yaml --model experiments/ITOP-SIDE/1/log/model.pth
```

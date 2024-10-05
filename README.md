# Pix2PixHD: Edges to Faces - CelebA-HQ

![Header Image](insert_image_link_here)  
> *Generating photorealistic faces from edge maps using Conditional GANs (Pix2PixHD)*

## 🌟 Overview
Welcome to the **Pix2PixHD: Edges to Faces** project! In this repository, I’ve implemented and trained a modified **Pix2PixHD** model to generate high-resolution (1024x1024) photorealistic face images from edge maps using the **CelebA-HQ** dataset. This project extends the groundbreaking work presented in [“High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs”](https://arxiv.org/abs/1711.11585), providing an exciting tool for tasks such as sketch-based image synthesis and image editing.

## 🚀 Results
The model takes edge maps as input and generates stunning, realistic face images. Below are some results from Phase 1 of training:

| Input (Edge Map) | Output (Generated Image) |
|:----------------:|:-----------------------:|
| ![epoch_3_step_95800_img_1_sketch](https://github.com/user-attachments/assets/076678db-3f47-4e2a-a194-756126ff2df8) | ![epoch_3_step_95800_img_1_generated](https://github.com/user-attachments/assets/035fbd57-36a8-493c-a667-52b83ef3c9f5) |
| ![epoch_3_step_97000_img_1_sketch](https://github.com/user-attachments/assets/89f19ea0-bfd9-4e58-b49d-2284c05e58fb) |  ![epoch_3_step_97000_img_1_generated](https://github.com/user-attachments/assets/c5e6d2e1-ff9b-461d-b275-56e08edbaaef) |

These outputs are produced using **1024x1024** resolution images, giving exceptional detail and realism!

## 📄 Original Paper
This implementation is based on the paper [“High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs”](https://arxiv.org/abs/1711.11585) by Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, and others. Pix2PixHD builds on the standard Pix2Pix architecture to produce images with higher resolutions and fidelity, especially useful for tasks such as semantic image synthesis.

## 📂 Project Structure
- **`/data`**: Contains the dataset preprocessing scripts and edge map generation using Canny edge detection.
- **`./`**: The core PyTorch implementation of Pix2PixHD, including Global Generator, Local Enhancer, and the multi-scale Discriminators.
- **`/output`**: The generated outputs from the model.
- **`train.py`**: Script for training the model on edge-to-image task.

## ⚙️ How to Run

### Prerequisites
To get started, make sure you have the following installed:
- Python 3.12.x
- PyTorch 2.4.x with CUDA

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/inventwithdean/pix2pixhd.git
    cd pix2pixhd
    ```
2. Download the **CelebA-HQ** dataset and place it in the `/data` folder. You can find it [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
3. Make sure you have all the faces inside `data/faces` folder. And create a directory `data/sketches`.
4. Generating edge maps:
    ```bash
    python create_dataset.py
    ```

### Training

To train the model on the edge-to-face task, run:
```bash
python train.py
```
Create directories `enhanced_checkpoints` and `global_checkpoints` for checkpointing.

## 📊 Features
- **High-Resolution Synthesis**: Generates 1024x1024 images from edge maps.
- **Conditional GAN Architecture**: Uses multi-scale discriminators for improved realism.
- **Instance-Level Control (WIP)**: Planned updates for fine-tuning specific features in generated images.
  
## 📝 Future Work
- Instance-wise control for fine-grained editing of facial features.
- Further improvements to training strategies for faster convergence.
- Integration of new datasets for generalization beyond CelebA-HQ.

## 🛠️ Contributing
Feel free to fork this repo, submit issues, or make pull requests. Contributions are always welcome to improve the codebase!

## 📬 Contact
If you have any questions or feedback, feel free to reach out to me via [LinkedIn](https://www.linkedin.com/in/inventwithdean) or open an issue on GitHub.

## 💡 Acknowledgements
This work is inspired by the original **Pix2PixHD** implementation by [NVIDIA](https://github.com/NVIDIA/pix2pixHD) and the creators of the **CelebA-HQ** dataset.

---

Enjoy experimenting! If you like the project, don’t forget to give it a ⭐ on GitHub!

# Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption And Poisoning Against Deepfake

![License](https://img.shields.io/badge/License-Custom-blue)
![Python Version](https://img.shields.io/badge/Python-3.6+-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![Status](https://img.shields.io/badge/Code%20Status-Coming%20Soon-red)

<br>

> **Important Notice**
>
> This is the official PyTorch implementation for the paper "Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption And Poisoning Against Deepfake."
>
> **The code is currently undergoing final cleanup and review and will be released publicly soon. Stay tuned!**

---

This framework introduces a robust solution addressing the limitations of existing active defenses against Deepfake manipulation. By generating a carefully optimized, imperceptible watermark, this framework operates on a two-stage defense principle of "Interruption" and "Poisoning." It not only degrades the quality of immediate forgery attempts but also critically undermines an attacker's ability to mount adaptive attacks via retraining.

## Table of Contents
- [Key Features](#key-features)
- [Framework Overview](#framework-overview)
- [Requirements](#requirements)
- [Installation (Available After Code Release)](#installation-available-after-code-release)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Stage 1: Training the Interruption Perturbation](#stage-1-training-the-interruption-perturbation)
  - [Stage 2: Training the Poisoning Perturbation](#stage-2-training-the-poisoning-perturbation)
- [Results](#results)
- [How to Cite](#how-to-cite)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact Us](#contact-us)

## Key Features

* **Two-Stage Active Defense**: Innovatively combines "Immediate Interruption" and "Delayed Poisoning" to provide both real-time and long-term protection.
* **Imperceptible Watermark**: The generated protective perturbation is visually imperceptible, ensuring the usability and aesthetic quality of the original images.
* **Resilience to Adaptive Attacks**: By corrupting the training process with poisoned data, it effectively prevents attackers from using protected images to retrain or fine-tune their generative models.
* **Broad Effectiveness**: The framework's efficacy has been demonstrated through comprehensive evaluations against an ensemble of widely-used generative models, such as StarGAN, AttentionGAN, HiSD, and AttGAN.

## Framework Overview

Our defense framework is composed of two core components: an **Interruption Module** and a **Poisoning Module**.

1.  **Interruption Stage**: Applies a subtle perturbation to the original image. This perturbation is sufficient to significantly disrupt the structure and quality of the output when a Deepfake model attempts a face-editing or swapping task.
2.  **Poisoning Stage**: Building upon the interruption perturbation, a poisoning component is added. When an attacker collects these protected images for training their own model, this poisoned data contaminates their training set, leading to severe performance degradation or complete model failure.

```
[Framework Architecture Diagram Will Be Added Here]

 Original Image --> [Interruption Module] --> Protected Image --> [Deepfake Model (Immediate Attack)] --> Degraded Output
                               |
                               +--> [Poisoning Module] --> Poisoned & Protected Image --> [Attacker's Retraining Dataset] --> Failed/Corrupted Model
```

## Requirements

* **Python**: 3.6+
* **PyTorch**: 1.9+
* **GPU**: A CUDA-capable GPU is recommended.
* See the `requirements.txt` file for a complete list of dependencies.

## Installation (Available After Code Release)

1.  **Clone the repository**
    ```bash
    # The repository URL will be provided upon code release
    git clone [https://github.com/TSDF_code](https://github.com/TSDF_code)
    cd TSDF_code
    ```
   

2.  **Install dependencies**
    We recommend using a virtual environment (e.g., `conda`) to avoid package conflicts.
    ```bash
    pip install -r requirements.txt
    ```
   

## Usage

### Dataset Preparation

This project uses the CelebA dataset. Please follow the steps below to download and prepare it:

```bash
cd stargan
bash download.sh celeba
```

After downloading, ensure that the `stargan/data/celeba` directory contains the `img_align_celeba` folder and the `list_attr_celeba.txt` file.

### Stage 1: Training the Interruption Perturbation

The goal of this stage is to train the interruption component that can instantly disrupt Deepfake outputs.

* **Configuration**: In the training script or a config file, set `do_poison = False`.
* **Training Steps**: Set the desired number of training steps (e.g., `interruption_steps`).

```bash
# Ensure do_poison is set to False in the configuration
python train.py
```


### Stage 2: Training the Poisoning Perturbation

After obtaining an optimized interruption perturbation, this stage trains the data poisoning component for persistent defense.

* **Configuration**: In the training script or a config file, set `do_poison = True`.
* **Training Steps**: Set the desired number of training steps (e.g., `poisoned_steps`).

```bash
# Ensure do_poison is set to True in the configuration
python train.py
```


## Results

Upon the official release of the code, we will present qualitative and quantitative results here. This will include comparisons against baseline methods and visualizations of the defense effectiveness on various generative models to demonstrate the robustness of our framework.

## How to Cite

If you use the code or ideas from this project in your research, please cite our paper:

```bibtex
@article{Zheng2025Boosting,
  title={Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption And Poisoning Against Deepfake},
  author={Hongrui Zheng, Yuezun Li, Liejun Wang, Yunfeng Diao, Zhiqing Guo},
  journal={arxiv},
  year={2025}
}
```


## License

This project is licensed under the following terms:
* It is free for **academic research purposes**.
* Any **commercial use** requires prior authorization.

For commercial licensing, please contact us via the email below.

## Acknowledgements

We thank the researchers in this field for their outstanding work. The implementation of this project also drew inspiration from several excellent open-source projects, which will be credited in the source code.

## Contact Us

If you have any questions about this project or would like to collaborate, please feel free to contact us:

* **Email**: `107552301310@stu.xju.edu.cn`

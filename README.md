# COMP34812 Coursework Repository

This repository contains the deliverables for the COMP34812 coursework on natural language inference. It includes implementations, evaluation scripts, training notebooks, and model cards for reproducibility.

---

## Task Description

**Task A:**  
Given a premise and a hypothesis, determine whether the hypothesis is true based on the premise. The training set consists of 26K premise-hypothesis pairs, and the validation set comprises over 6K pairs.

---

## Solutions

### **Solution B: Deep Learning without Full Transformer Architectures**

Our final model leverages a decomposable attention architecture combined with a pre-trained DeBERTa-v3 transformer embedding layer, following the approach introduced by Parikh et al. (2016). It employs three key steps—**Attend**, **Compare**, and **Aggregate**—to efficiently align, compare, and integrate subphrase information in parallel, achieving strong results while using fewer parameters.

- **Embeddings Repository (DeBERTa-v3):**  
  [https://huggingface.co/microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)

### **Solution C: Transformer-Based Approach**

Our solution is a natural language inference classifier built on a pretrained RoBERTa backbone that determines whether a hypothesis is entailed by a given premise. It employs a multi-stage fine-tuning strategy to capture subtle semantic relationships.

- **RoBERTa Large Repository:**  
  [https://huggingface.co/FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)

---

## Repository Structure

- **Demo Code Folder:**  
  Contains the demo code for the flash presentation where the models are loaded to generate predictions on the test set.

- **Model Evaluation Folder:**  
  Provides evaluations on the models via traditional classification metrics and visualizations such as the Confusion Matrix, ROC Curve, and Precision-Recall Curve, including additional evaluation on an augmented dataset.

- **Model Cards Folder:**  
  Offers detailed documentation about the models, source papers, and resources for reproducibility.

- **Model Training Folder:**  
  Contains the Jupyter notebooks used to train the models.

---

## Model Links

- **Solution B Model:**  
  [Model B on Google Drive](https://drive.google.com/drive/folders/1qKPlkRNBiQqxplKhBZ2VYT1ZZRrpiKeV?usp=sharing)

- **Solution C Model:**  
  [Model C on Google Drive](https://drive.google.com/drive/folders/1-rDdvSoXzxbJ95xbaAU0vDrusOoWGrXO?usp=sharing)

---

## Instructions to Run

1. **Datasets and Models:**  
   - If using Google Drive or Colab, download the datasets and models to your drive.
   - For other environments, ensure you have the required files locally.

2. **Dependencies:**  
   Refer to the model cards for the list of necessary libraries and packages for each implementation.

3. **Google Colab Specifics:**  
   If you are not using the Google Colab environment, comment out the following two lines at the top of the first code cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

## Sources and References

- **Universal Language Model Fine-Tuning (ULMFiT):**  
  Howard, Jeremy, and Sebastian Ruder. “Universal Language Model Fine-Tuning for Text Classification.” [arXiv:1801.06146](https://arxiv.org/abs/1801.06146), 2018.

- **Context-Aware Legal Citation Recommendation Using Deep Learning:**  
  Huang, Zihan, et al. “Context-Aware Legal Citation Recommendation Using Deep Learning.” [PDF](https://arxiv.org/pdf/2106.10776.pdf) | [DOI](https://doi.org/10.1145/3462757.3466066), Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law, 2021.

- **Stochastic Weight Averaging for Improved Generalization:**  
  Izmailov, Pavel, et al. “Averaging Weights Leads to Wider Optima and Better Generalization.” [arXiv:1803.05407](https://arxiv.org/abs/1803.05407), 2019.

- **Decomposable Attention Model for Natural Language Inference:**  
  Parikh, Ankur P., et al. “A Decomposable Attention Model for Natural Language Inference.” [arXiv:1606.01933](https://arxiv.org/abs/1606.01933), 2016.

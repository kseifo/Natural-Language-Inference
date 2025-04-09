---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/kseifo/Natural-Language-Inference

---

# Model Card for t94363ha-g64462ks-NLI

<!-- Provide a quick summary of what the model is/does. -->

The model is a natural language inference classifier designed to determine whether a hypothesis is entailed by a given premise. It uses a pretrained RoBERTa as its backbone, which provides rich contextual embeddings, and is fine-tuned using a strategy inspired by ULMFiT and stochastic weight averaging (SWA). Initially, training focuses solely on the classification head, allowing the network to rapidly adapt to the specific nuances of the NLI task. Following this, a gradual unfreezing (ULMFiT approach) of the encoder layers is implemented, with discriminative learning rates applied to ensure that lower layers are updated more conservatively compared to higher, task-specific layers. In the final phase, the entire model undergoes full fine-tuning with stochastic weight averaging (SWA), which averages the weights over successive epochs to smooth out the optimization process and enhance generalization. This multi-stage training approach enables the classifier to effectively capture subtle semantic relationships between premises and hypotheses.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a RoBERTa model that was fine-tuned using a ULMFiT and SWA inspired strategy
trained on more than 24K premise-hypothesis pairs.

- **Developed by:** Hala Alsaffarini and Kareem Seifo
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model:** roberta-large

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-large
- **Paper or documentation:**
  - https://arxiv.org/abs/2101.04965
  - https://arxiv.org/abs/1801.06146
  - https://arxiv.org/pdf/2212.05956


## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The training data consists of 24,432 pairs of texts provided by the COMP34812 Team, with each pair including a hypothesis, a premise, and a prediction column. The prediction column indicates whether the hypothesis is entailed by the premise.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


    Phase 1 (Classifier Head) Hyperparameters:
      - learning_rate: 5e-5
      - weight_decay: 0.01
      - train_batch_size: 8
      - eval_batch_size: 8
      - seed: 42
      - num_epochs: 1
      
    Phase 2 (Gradual Unfreezing) Hyperparameters:
      - learning_rate: 5e-5 * (0.9 ** (24 - ith_layer))
      - weight_decay: 0.01
      - train_batch_size: 8
      - eval_batch_size: 8
      - seed: 42
      - num_epochs_per_layer: 1

    Phase 3 (Overall SWA Tuning) Hyperparameters:
      - learning_rate: 2e-5
      - weight_decay: 0.01
      - train_batch_size: 8
      - eval_batch_size: 8
      - seed: 42
      - num_epochs: 2
      - SWA Settings: SWA averaging starts at 50% of the total training steps, with weight updates every 10 steps
      

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 7 hours (with Hyperparameter Tuning), 1 hour (without Hyperparameter Tuning)
      - duration per training epoch: 3 mins (Phase 1), 5 mins (Phase 2), 18 mins (Phase 3)
      - model size: 1355MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A development set amounting to 6,736 pairs of texts provided by the COMP34812 Team, with each pair including a hypothesis, a premise, and a prediction column.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Micro Precision
      - Macro Recall
      - Macro F1 
      - Weighted Precision
      - Weighted Macro Precision
      - Weighted F1
      - Matthews Correlation Coefficient (MCC)
      - Accuracy
      

### Results

The model obtained an accuracy of 92.01%, a macro precision of 92.04%, a macro recall of 91.98%, a macro F1 of 91.99%, a weighted macro precision of 92.02%, a weighted macro recall of 92.01%, a weighted macro F1 of 92.01%, and a Matthews correlation coefficient of 0.84.

### Additional Evaluation
An additional evaluation was conducted on an augmented version of the development dataset. This augmented dataset was created by applying synonym replacement to the premise and hypothesis fields of each sample with a 25% probability per eligible word. Common conjunctions, short words, and punctuation were excluded from replacement. For each original example, one or two augmented variants were generated by selectively modifying either the premise, the hypothesis, or both. This approach aimed to assess the model's robustness to lexical variation and ensure its performance remains consistent when evaluated on semantically equivalent, yet lexically altered, inputs.

#### Results on Augmented Dataset
The model achieved an overall accuracy of 87.49% on the augmented dataset. Class-wise, it attained a precision of 0.88, recall of 0.86, and F1-score of 0.87 for class 0, while for class 1, the precision was 0.87, recall 0.89, and F1-score 0.88.

## Technical Specifications

### Minimum Hardware Requirements

      - RAM: at least 16 GB
      - Storage: at least 3GB
      - GPU: V100

### Hardware Used (Colab Pro Subscription)
      - RAM: 40 GB
      - Storage: 235 GB
      - GPU: A100

### Software


      - Transformers (>=4.18.0)
      - Datasets (>=1.18.3)
      - scikit-learn (>=0.24.2)
      - pandas (>=1.3.5)
      - numpy (>=1.21.2)
      - PyTorch (>=1.11.0+cu113)
      - safetensors (latest)
      - Google Colab (for drive integration)
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Input Length:**  
  Any inputs (concatenation of two sequences) longer than 128 subwords will be truncated by the model.

- **Handling Class Imbalance:**  
  If you decide to re-train the model on a different dataset, make sure to use the `calculate_normalized_class_weights()` function to compute class weights. This will help address any imbalances in the dataset by ensuring that underrepresented classes receive appropriate emphasis during training.


## Additional Information

<!-- Any other information that would be useful for other people to know. -->


  - The hyperparameters were determined by a bayesian search of 20 trials.
  - The run times are based on the A100 GPU with 40 GB RAM (Google ColabPro).
    
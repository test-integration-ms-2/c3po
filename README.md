# RLVF: Learning from Verbal Feedback without Overgeneralization

## Overview
This repository provides a reference implementation of Contextualized Critiques with Constrained Preference Optimization (C3PO), a novel technique for aligning Large Language Models (LLMs) with user preferences. C3PO allows models to learn from a single sentence of feedback without overgeneralizing beyond the intended context. The repository includes implementations of C3PO, relevant baselines, and all components necessary to explore the techniques proposed in our paper.

For more information, visit the [project website](https://austrian-code-wizard.github.io/c3po-website/).

Paper: [RLVF: Learning from Verbal Feedback without Overgeneralization](https://arxiv.org/abs/2402.10893)

## Repository Structure
- `src/feedback/`: Contains feedback definitions for C3PO. You can create your own by following the format in the provided files. The feedbacks used in our paper experiments are in `src/feedback/final_exp.py`.
- `src/modal/`: Contains all files related to launching sampling, training, and evaluation jobs on Modal.
- `src/lcdpo.py`: Implements the custom trainer using the C3PO loss function.
- `src/sft_weighted.py`: Implements a modified SFT trainer that includes weighted samples of near-scope and out-of-scope data in the loss. This is used for the `SCD + Negatives` baseline.
- `src/configs/`: Contains configuration files for sampling, training, and evaluating using the various methods described in the paper.
- `src/sample.py`: Handles data sampling for training and evaluation.
- `src/train.py`: Implements the training pipeline.
- `src/eval.py`: Implements the evaluation pipeline.

Other files in the repository should be identifiable based on their naming conventions.

# Running C3PO
## Prerequisites
- Ensure you have Conda/Miniconda installed
- Create a conda environment: `conda create -n gfl python=3.11`
- Activate the environment: `conda activate gfl`
- Install dependencies: `pip install -r requirements.txt`

## Supported Platforms
We utilize [Modal](https://www.modal.com) as our primary compute provider, and the examples below assume you have created and configured a Modal account. However, all functionality will work with any compute setup that provides sufficient resources:
- For training: We recommend a 40GB A100 GPU
- For evaluation: We recommend an L4 or A10G GPU

### Running Locally vs. on Modal
- **Modal**: Use the commands as shown in the examples below
- **Local**: Replace `modal run src.modal.app` with `python <script>.py` where `<script>.py` can be one of:
  - `src/sample.py`
  - `src/train.py`
  - `src/eval.py`
   
Note: When running locally, only the `--arg_file`, `--run_id`, `--data_dir`, and `--feedback_prefix` flags are currently supported.

## C3PO Workflow
C3PO operates in a three-step workflow: Sampling → Training → Evaluation. Each step builds on the previous one and should use the same `run-id` to maintain data consistency.

### 1. Sampling
Sampling is the first step of C3PO. This process:
- Identifies categories relevant to each feedback specified in your command line arguments
- Generates prompts for each category
- Creates baseline, revised, in-context, and Chain-of-Thought (CoT) completions

**Example:**
```bash
modal run src.modal.app --arg-file configs/config.json --do-sample --feedback-prefix "Be more detailed" --run-id test
```

### 2. Training
The training step fine-tunes the specified base model on each of the feedbacks passed. This step uses the data generated during the sampling phase.

**Example:**
```bash
modal run src.modal.app --arg-file configs/config.json --do-train --feedback-prefix "Be more detailed" --run-id test
```

### 3. Evaluation
The evaluation step compares responses from the method specified in the eval section of your config file with the baseline responses. This helps measure the effectiveness of your fine-tuned model.

**Example:**
```bash
modal run src.modal.app --arg-file configs/config.json --do-eval --feedback-prefix "Be more detailed" --run-id test
```

## Command Line Arguments
You can pass the following arguments to `modal run src.modal.app`:

### Required Arguments
- `--arg_file`: File path to the config file to use.

### Workflow Control
- `--do_sample`: Flag to indicate whether to sample data. Defaults to `False`.
- `--do_train`: Flag to indicate whether to train the model. Defaults to `False`.
- `--do_eval`: Flag to indicate whether to evaluate the model. Defaults to `False`.

### Data Management
- `--run_id`: ID of the current run. Sampling/training/eval on the same data must use the same run_id.
- `--data_dir`: Path to the output data directory. Defaults to `/results/data` for Modal and `./data` for local.
- `--copy_results`: Flag to indicate whether to copy results to local machine. Defaults to `False`.

### Feedback Selection
- `--feedback_prefix`: Allows filtering feedbacks by prefix. Defaults to `None`.
- `--second_feedback_prefix`: Allows specifying a second feedback for combined data training or combined adapter evaluation using its prefix. There must only be one feedback with this prefix. Defaults to `None`.
- `--feedback_category`: Allows filtering feedbacks by category. Defaults to `None`.

### Hyperparameter Tuning
- `--sweep_params`: List of config parameter names to sweep. For example, `["training_args.learning_rate", "training_args.lora_r"]`. Defaults to `None`.
- `--sweep_values`: List of tuples of values to sweep. For example, `[(1e-5, 0.05), (5e-5, 0.1)]`. Defaults to `None`.

## Important Notes
- **Data Storage**: Sampling and evaluation data will be stored on the remote Modal volume and automatically copied to your local machine after each run, preserving the file system structure.
- **Configuration Options**: Check out the Sampling, Training, and Model arguments in the configuration files for a full overview of all possible options.
- **Overwriting Warning**: By default, training and evaluation runs with the same parameters and `run_id` will overwrite previous runs. Be careful to use unique `run_id` values when you want to preserve previous results.

## Citation

If you use this code, please cite our paper.

```
@misc{stephan2024rlvf,
      title={RLVF: Learning from Verbal Feedback without Overgeneralization}, 
      author={Moritz Stephan and Alexander Khazatsky and Eric Mitchell and Annie S Chen and Sheryl Hsu and Archit Sharma and Chelsea Finn},
      year={2024},
      eprint={2402.10893},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Deep Learning for Sequence-to-Sequence Transliteration

This project implements and evaluates various Recurrent Neural Network (RNN) based models for the task of transliteration, specifically converting Hindi words from Latin script (Romanized) to their native Devanagari script. The project explores vanilla RNNs, LSTMs, GRUs, and attention mechanisms. Hyperparameter tuning is performed using Weights & Biases (wandb), and models are evaluated on the Dakshina dataset.

## Project Structure

The project is organized into three main Jupyter Notebooks: <br>
In the first 2 notebooks, there are 2 options - 1) Do sweep over hyperparameters, 2) or directly test the best configuration of paramters on test data. 

1.  **`Q1-Q2-Q3-Q4.ipynb`**:
    *   **Question 1**: Theoretical calculations for the number of computations and parameters in a basic RNN seq2seq model.
    *   **Question 2**: Implementation of a vanilla seq2seq model (configurable with RNN, LSTM, GRU cells). Hyperparameter tuning using Weights & Biases sweeps to find the best configuration for Hindi transliteration.
    *   **Question 3**: Analysis of the wandb sweep plots from Question 2 to derive insights about hyperparameter impact on model performance.
    *   **Question 4**: Evaluation of the best vanilla seq2seq model (from Q2) on the test set. Includes accuracy reporting, sample predictions, and error analysis. Predictions are saved in `predictions_vanilla/test_predictions_vanilla.tsv`.

2.  **`Q5.ipynb`**:
    *   **Question 5**: Implementation of a seq2seq model with a Bahdanau-style attention mechanism.
        *   **(a)**: Hyperparameter tuning for the attention model using wandb sweeps.
        *   **(b)**: Evaluation of the best attention model on the test set and reporting accuracy. Predictions are saved in `predictions_attention/test_predictions_attention.tsv`.
        *   **(c)**: Comparison of the attention model's performance against the vanilla model, including analysis of corrected/new errors.
        *   **(d)**: Generation and visualization of attention heatmaps for selected test samples to understand where the model focuses during decoding.

3.  **`Q6.ipynb`**:
    *   **Question 6 (Challenge)**: Implementation of a method to visualize the "connectivity" or "pseudo-attention" in the vanilla seq2seq model (from Q1-Q4). This involves:
        *   Retraining the best vanilla model.
        *   Building specialized inference models to extract encoder and decoder hidden states.
        *   Calculating similarity scores (e.g., dot product) between decoder states and all encoder states for each output step.
        *   Normalizing these scores (softmax) to create a pseudo-attention heatmap, showing which input characters the decoder might be "looking at" when generating an output character.
        *   Plotting these heatmaps for selected test samples.

## Dataset

The project uses the **Dakshina Dataset** (specifically the Hindi `hi` portion from `dakshina_dataset_v1.0/hi/lexicons/`). This dataset provides pairs of words in their native Devanagari script and their Romanized (Latin script) versions.
*   The data is split into `train`, `dev` (validation), and `test` sets (e.g., `hi.translit.sampled.train.tsv`).
*   Place the `dakshina_dataset_v1.0` directory in the root of the project folder.

## Core Functionality

*   **Data Loading and Preprocessing**: Functions to load TSV data, create character-level vocabularies for input (Roman) and target (Devanagari) scripts, and vectorize sequences (padding, token-to-index mapping).
*   **Model Architectures**:
    *   Configurable vanilla sequence-to-sequence (encoder-decoder) model with options for RNN, LSTM, or GRU cells, and variable layers/hidden sizes.
    *   Sequence-to-sequence model with a Bahdanau-style attention mechanism.
*   **Training**: Models are trained using teacher forcing with categorical cross-entropy loss. Early stopping is used to prevent overfitting.
*   **Inference**:
    *   Specialized encoder and decoder models are built for inference.
    *   Beam search decoding is implemented for generating predictions.
*   **Hyperparameter Tuning**: Weights & Biases (wandb) sweeps are used to find optimal hyperparameters (e.g., embedding size, hidden size, cell type, dropout, learning rate, optimizer).
*   **Evaluation**: Models are evaluated based on exact match accuracy on the test set.
*   **Visualization**:
    *   Attention heatmaps for the attention-based model (Q5).
    *   Pseudo-attention heatmaps for the vanilla model to infer input-output alignments (Q6).

## Setup and Requirements

1.  **Python Environment**: Python 3.9+ is recommended.
2.  **Libraries**: Install the required Python libraries:
    ```
    pip install numpy pandas tensorflow scikit-learn wandb matplotlib ipython
    ```
3.  **Dataset**: Download the Dakshina dataset. Extract it and ensure the `dakshina_dataset_v1.0` directory is in the root of your project.
4.  **Weights & Biases Account**:
    *   Sign up for a free account at [wandb.ai](https://wandb.ai).
    *   Log in to wandb in your environment using the CLI:
        ```
        wandb login
        ```
    *   The notebooks will prompt for login if not already authenticated.
    *   Ensure you update the `entity` and `project` names in the `wandb.sweep()` and `wandb.init()` calls within the notebooks if you are using your own W&B account. The current entity is set to `ce21b097-indian-institute-of-technology-madras`.

## Running the Code

The project is divided into three Jupyter Notebooks, corresponding to different parts of the assignment. It's recommended to run them sequentially or as per the assignment questions they address.

### General Instructions for Running Notebooks:

1.  **Kernel**: Ensure you are using a Python 3 kernel in Jupyter.
2.  **GPU**: For training deep learning models, using a GPU is highly recommended. If running locally, ensure TensorFlow is configured for GPU usage (CUDA, cuDNN installed and compatible). If using Google Colab, enable the GPU runtime (Runtime > Change runtime type > Hardware accelerator > GPU).
3.  **Execute Cells Sequentially**: Run the cells in each notebook from top to bottom. Many cells depend on variables and functions defined in previous cells.
4.  **File Paths**: Verify that the `dataset_base_dir` variable in the data loading cells correctly points to the location of your `dakshina_dataset_v1.0` folder.
5.  **Wandb Sweeps**:
    *   Cells that initiate `wandb.sweep()` will define a sweep configuration and start a sweep controller.
    *   The subsequent `wandb.agent()` call will start an agent that picks hyperparameter combinations and runs the training function.
    *   You can run multiple agents in parallel if you have the compute resources.
    *   Monitor sweep progress on your W&B dashboard.
    *   The `count` parameter in `wandb.agent(..., count=N)` specifies how many runs the agent will perform. Adjust this based on your compute budget.

### Specific Notebook Instructions:

**1. `Q1-Q2-Q3-Q4.ipynb` (Vanilla Seq2Seq Model)**

*   **Cell Execution**: Run all cells.
*   **Question 2 (Sweep)**:
    *   The sweep configuration is defined in a cell.
    *   The `wandb.agent()` cell will execute the hyperparameter search. This can take a considerable amount of time depending on the `count` and complexity of models.
    *   After the sweep, analyze the plots on your W&B dashboard to answer Question 3.
*   **Question 4 (Test Evaluation)**:
    *   Update the `best_config_dict` with the best hyperparameters found from your sweep in Question 2.
    *   This section will retrain the model with the best configuration on the full training data and then evaluate it on the test set.
    *   Predictions will be saved to `predictions_vanilla/test_predictions_vanilla.tsv`.

**2. `Q5.ipynb` (Attention Model)**

*   **Cell Execution**: Run all cells.
*   **Question 5a (Attention Sweep)**:
    *   A new sweep configuration for the attention model is defined.
    *   The `wandb.agent()` cell will run the sweep. This will also take time.
    *   Analyze plots on W&B for your report.
*   **Question 5b, 5c, 5d (Attention Test Evaluation & Heatmaps)**:
    *   Update the `best_config_attention_dict` with the best hyperparameters from the attention model sweep.
    *   This section retrains the best attention model, evaluates it on the test set, and saves predictions to `predictions_attention/test_predictions_attention.tsv`.
    *   It includes a function `plot_attention_heatmap` for visualizing attention. The final cell in the notebook (or a new one you add) should use this function to generate and display heatmaps for selected test samples, typically in a 3x3 grid using `matplotlib.pyplot.subplots`.

**3. `Q6.ipynb` (Vanilla Model "Connectivity" Visualization)**

*   **Cell Execution**: Run all cells.
*   **Best Vanilla Config**: Update the `best_vanilla_config_dict` at the beginning of this notebook with the same best configuration used in Question 4 (from the `Q1-Q2-Q3-Q4.ipynb` sweep).
*   **Model Retraining**: The notebook first retrains this best vanilla model.
*   **Inference for Visualization**: Specialized inference models are built to extract intermediate hidden states.
*   **Pseudo-Attention Calculation**: A function calculates similarity scores between decoder and encoder states.
*   **Plotting**: The final cell generates and displays pseudo-attention heatmaps for selected test samples in a 3x3 grid.
    *   **Font Issues**: The notebook output shows warnings like `UserWarning: Glyph XXXX missing from font(s) DejaVu Sans.` and `UserWarning: Matplotlib currently does not support Devanagari natively.`. This means the default Matplotlib font does not have Devanagari characters. To fix this for clearer visualizations, you might need to install a Devanagari-supporting font (e.g., "Nirmala UI", "Mangal", or other Unicode Devanagari fonts) on your system and configure Matplotlib to use it:
        ```
        # Example: Add to the import cell or before plotting
        import matplotlib.font_manager as fm
        # plt.rcParams['font.family'] = 'Nirmala UI' # Or another Devanagari font available
        # Or find and specify path to a .ttf font file
        # font_path = 'path/to/your/devanagari_font.ttf'
        # if os.path.exists(font_path):
        #     custom_font = fm.FontProperties(fname=font_path)
        #     plt.rcParams['font.family'] = custom_font.get_name()
        # else:
        #     print(f"Font not found at {font_path}, Devanagari characters may not render correctly.")
        ```
        This might require some experimentation based on your OS and available fonts.

## Expected Outputs

*   **W&B Reports**: Links to W&B sweep reports demonstrating hyperparameter tuning for both vanilla and attention models.
*   **Prediction Files**:
    *   `predictions_vanilla/test_predictions_vanilla.tsv`
    *   `predictions_attention/test_predictions_attention.tsv`
*   **Visualizations**:
    *   Attention heatmaps from `Q5.ipynb`.
    *   Pseudo-attention heatmaps from `Q6.ipynb`.
*   **Jupyter Notebooks**: Completed notebooks (`Q1-Q2-Q3-Q4.ipynb`, `Q5.ipynb`, `Q6.ipynb`) with code, outputs, and analyses.

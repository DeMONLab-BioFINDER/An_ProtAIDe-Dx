# Example

---

## Setup

---

### Conda environment

---

We recommend using `conda` to create a virtual environment for running the example code. You can create and activate the environment by running:

```bash
conda env create -f replication/config/DeMONLab_ProtAIDe.yaml -n ProtAIDe
conda activate ProtAIDe
```

### Simulated data

---

For example usage, we provide a simulated proteomics dataset (N = 120; `data/Simulated_SomaLogic_120Subjects.csv`). The dataset includes 648 proteins (selected as in our paper), clinical diagnoses for six conditions, demographic variables, and biomarker measurements.

Because the dataset is simulated, results (e.g, classification performances) may differ from those reported in our paper. Its primary purpose is to illustrate how to use the pre-trained models and to help users become familiar with the workflow. Once you are comfortable with the pipeline, we recommend applying the method to your own data.

### Users' own data

---

Users are welcome to apply our pre-trained ProtAIDe-Dx models to their own proteomics datasets. Before doing so, please ensure that your data are preprocessed in a manner consistent with the training data used in our study, including normalization, missing-value handling, and any other relevant steps as described in our code.

Please also note that ProtAIDe-Dx was trained on `SomaLogic 7k` `plasma proteomics`. If your data come from a different platform (e.g., `Olink`), a different biofluid (e.g., `CSF` or `serum`), or a different SomaLogic panel/version (e.g., `SomaLogic 11k`), we recommend performing an bridging/harmonization procedure and validating performance before interpretation to ensure the model is being applied appropriately.

## Notebook example

---

In our `example/ExampleUsage.ipynb` notebook, we provide a step-by-step guide on how to apply the pre-trained ProtAIDe-Dx models on the simulated dataset. This notebook covers:

1. data preprocessing;
2. disease classification;
3. individual probability visualization;
4. SHAP analysis for model interpretation;
5. Correlate biomarkers with model embeddings or predicted probabilities.

## Clean up

---

Once you have finished the replication, you can clean up the generated results by running:

```bash
bash example/scripts/clean_examples.sh
```

## Bugs and questions

---

Please contact Lijun An at anlijuncn@gmail.com for any questions or issues.

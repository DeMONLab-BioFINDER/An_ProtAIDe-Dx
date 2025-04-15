# Benchmarking the AI-based diagnostic potential of plasma proteomics for neurodegenerative disease in 17,170 people 
---
## Background
The rapid growth of the dementia population emphasizes the urgent need for accessible and scalable biomarkers for neurodegenerative diseases. Such biomarkers are sorely needed to enable enhanced early diagnosis, confirmation of (co-)pathologies, and improved participant selection in clinical trials. This highlights the need for a one-shot, multi-disease biomarker panel that is minimally invasive, cost-effective, and widely accessible. In this paper, we develop such a panel by applying state-of-the-art AI approaches to complex, high-dimensional proteomics data sourced from the GNPC, by far the largest neurodegenerative disease proteomics sample to date (n=17,170). 





![main_figures_from_paper](./ProtAIDe-Dx.jpg)

---

## Usage

### Environment setup

-   Our code uses Python and R, here is the setup:
    1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/#download-section) with Python 3.x if you don't have conda
    2. Create conda environment from our `replication/config/ProtAIDe-Dx_python_env.yml` file by `conda env create -f replication/config/ProtAIDe-Dx_python_env.yml`

### Example

-   The example of our code is detailed in `examples/README.md`

### Replication

-   If you have access to the [GNPC dataset](https://www.neuroproteome.org/), you can replicate our result following the instructions detailed in `replication/README.md`.


---

## Updates

-   Release v0.0.1 (XX/XX/2025): Initial release of ProtAIDe-Dx project

---

## Bugs and Questions

Please contact Lijun An at anlijuncn@gmail.com

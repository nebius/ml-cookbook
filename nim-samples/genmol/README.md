# NVIDIA GenMol

**The following code examples have been modified (on 16 September 2025) from the [NVIDIA digital biology examples](https://github.com/nvidia/digital-biology-examples/tree/main/examples/nims/genmol) to align with the Nebius deployment framework and facilitate customer integration. For the unmodified and complete examples, please refer to the original repository.**

## Molecule Generation with the GenMol NIM
GenMol is a masked diffusion model trained on molecular Sequential Attachment based Fragment Embeddings, or [SAFE representations](https://arxiv.org/abs/2310.10773) for fragment-based molecule generation.  GenMol a generalist model for various drug discovery tasks, including de novo generation, linker design, motif extension, scaffold decoration/morphing, hit generation, and lead optimization.  The use of the SAFE format allows for flexibility in the generation schema, such as:

 1. Specifying fixed fragment(s), which will remain unchanged in generation
 2. Specifying specific positions that generated fragments will attach to
 3. Generating a partial or full fragment or generating multiple fragments
 4. Generating fragments at any range of lengths specified.


The example notebooks in this repository demonstrate how to deploy and build these workflows with the NVIDIA GenMol NIM for fragment-based molecule generation:
 1. [GenMol Basics](1.basics.ipynb)
 2. [Linker Design](2.linder-design.ipynb)
 3. [Hit Generation](2.hit-generation.ipynb)

To run these examples locally, see the [Setup](#Setup) section below.

***

<img src="genmol-use-cases.png" alt="GenMol use cases" width="800"/>

NVIDIA BioNeMo NIMS can be integrated into existing workflows to leverage cutting edge Gen-AI capabilities for drug discovery, from ligand generation to protein folding to docking. These capabilities are also integrated into reference workflows in NVIDIA NIM Agent Blueprints. For more details, please visit [NVIDIA NIM Blueprints](https://build.nvidia.com/nim/blueprints).

## Setup

### Preparing the GenMol NIM

Install your GenMol NIM through Nebius Applications Catalog. You can verify that the NIM has started successfully by querying its status. 
After replacing the username, password, and URL in the following command to the exposed NIM endpoint, it will return `{"status":"ready"}` when the NIM is ready:

```bash
   curl -X 'GET' \
     'https://{username}:{password}@{url}/v1/health/ready' \
     -H 'accept: application/json'
```

### Jupyter Lab Environment

Prepare your JupyterLab environment locally, or use JupyterLab provided by Nebius Standalone Applications. 

### GenMol Notebook Dependencies and Launching the Lab Environment

Once the GenMol NIM and Jupyter Lab environment have been configured as above, clone this repository and make sure you install the requirements to run the notebooks

```bash
    pip install -r requirements.txt
```


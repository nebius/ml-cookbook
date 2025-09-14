# Copilot Instructions

## General Coding Guidelines

- All code generated must follow the official documentation and best practices for the relevant technology or framework.
- For Python code, adhere to [PEP8](https://peps.python.org/pep-0008/) style guidelines unless otherwise specified.
- Include clear and concise comments where necessary to explain non-obvious logic.
- Avoid deprecated APIs and prefer the latest stable features.
## Hugging Face (HF) Best Practices

- All Hugging Face code must follow the official [Hugging Face documentation](https://huggingface.co/docs) and reference implementations from the Hugging Face GitHub repositories.
- Use the latest stable APIs from `transformers`, `datasets`, and other Hugging Face libraries.
- Prefer using `AutoModel`, `AutoTokenizer`, and related `Auto*` classes for model and tokenizer instantiation.
- Load pretrained models and tokenizers using the `from_pretrained()` method.
- For training and evaluation, use the `Trainer` and `TrainingArguments` classes where appropriate, following official examples.
- When handling datasets, use the `datasets` library and its idiomatic patterns for loading, processing, and batching data.
- Ensure device-agnostic code by leveraging the `.to(device)` method and checking for available hardware (e.g., CUDA).
- When saving and loading models, use the recommended `save_pretrained()` and `from_pretrained()` methods.
- Reference the official Hugging Face documentation and example scripts for the most up-to-date and accurate patterns.
## PyTorch Specific Guidelines

- All PyTorch code must follow the official [PyTorch documentation](https://docs.pytorch.org/) and reference implementations from the [PyTorch GitHub repository](https://github.com/pytorch/pytorch) where applicable.
- Use idiomatic PyTorch patterns for model definition, training loops, and data loading.
- Prefer `torch.nn.Module` for model definitions and use `torch.utils.data.DataLoader` for batching data.
- Use device-agnostic code (e.g., `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- When saving and loading models, use the recommended `torch.save()` and `torch.load()` patterns.
- For distributed or multi-GPU training, follow the official [Distributed Training documentation](https://pytorch.org/docs/stable/distributed.html).

## Referencing Source Material

- When generating code, always check the official documentation and reference code for the latest and most accurate patterns.
- If a feature or pattern is not covered in the documentation, prefer examples from the official GitHub repository.
- Avoid using third-party or unofficial code snippets unless explicitly requested.

## Example

When asked to generate a PyTorch training loop, use the structure and patterns from:
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- https://github.com/pytorch/examples

## Updates

- Review and update these instructions regularly to reflect changes in official documentation or best practices.
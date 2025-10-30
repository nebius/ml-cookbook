# Boltz-2 Python Client Examples

**Here, we picked one sample from [NVIDIA digital biology examples](https://github.com/nvidia/digital-biology-examples/tree/main/examples/nims/boltz-2/examples) to show how to connect to Nebius NIM deployment. For the unmodified and complete examples, please refer to the original repository.**

## 🚀 Quick Start

### Pre-requisites
Install requirement using `requirements.txt`. This also includes the Boltz2 python client `boltz2-python-client`.

```bash
pip install -r requirements.txt
```

### Boltz2 NIM URL


#### Python API

The `boltz2_client` Python package assumes that your NIM instance is deployed either on localhost or on NVIDIA-hosted infrastructure. We have adjusted all the python samples here to work with the deployed Boltz2 NIM on Nebius. Go to [`constants.py`](constants.py) and replace the username, password, and the endpoint accordingly. 

```python
username = '<username>'
password = '<password>'
endpoint = '<endpoint>'
base_url = f'https://{username}:{password}@{endpoint}'
```

#### CLI with custom URL

Similarly, the `boltz2` CLI command also has the same assumption. To mitigate the issue, specify the URL to Boltz2 NIM deployed on Nebius as `--base-url` in the command line. For example, use the following command to check the health of the NIM deployment.

```bash
boltz2 --base-url <base_url> health 
```

## 📚 Additional Resources

- [**Original Boltz2 code samples module**](https://github.com/nvidia/digital-biology-examples/tree/main/examples/nims/boltz-2)
- **CLI Help**: `boltz2 --help`

## 🐛 Troubleshooting

1. **Service not running**: Ensure you point to the right URL of the deployed Boltz-2 NIM. Refer to [this section](#boltz2-nim-url).
2. **Timeout errors**: Increase `timeout` parameter for complex predictions
3. **Memory issues**: Reduce `diffusion_samples` or `sampling_steps`

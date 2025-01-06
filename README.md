# paper titel

## Conda Environment Configuration

To set up the required environment, follow these steps:

1. **Python Version**: 3.10.15
2. **Install Dependencies**:
    pytorch:2.1.1
    pytorch-cuda:12.1
    ```bash
    pip install -r requirements.txt
    ```
## Dataset generation

Follow the instructions of [LargeST](https://github.com/liuxu77/LargeST.git) to generate h5 files for SD, GBA, GLA, and CA, and then use our `generate_training_data.py` in the `data` folder to generate the dataset.

## Training

To train the model, run the following command:

```bash
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
  - `SD`
  - `GBA`
  - `GLA`
  - `CA`

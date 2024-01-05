# Text segmentation

## Enviroment set up

All of these enviroments assume that the device has a NVIDIA GPU. These enviroments have not been tested on a cpu only devices but should be able to run and compile. 

The preffered method is to use the Dev Container as this will ensure that the installed CUDA version matches what is specified in the packages. If you want to use the Dev Container via Visual Studio Code, [here](https://code.visualstudio.com/docs/devcontainers/containers) is a tutorial how to do so. 

If you have Mamba or Conda enviroment managers run from the project source directory:
```
mamba/conda env create -f ./environment.yml
```
This will create a new enviroment called `uda`, run `mamba/conda activate uda` to activate it.


If you want to use Python PIP package manager, create a Python 3.11 enviroment and run from the project source directory:
```
pip install -r ./requirements.txt
```

## Model training
To start training run `train.py`. All training arguments are CLI params and can be seen at the top of the script. Before running this script go through these params and adjust them according to your needs. Pay attention to `-data_path` and `-test_data_path`.

The data is split into two disjoint sets document wise for train and validation purposes. This means that no document appears in both sets.

All losses, metrics and visualisations are available through Tensorboard. Run `tensorboard --logidr ./{-project_dir}` to see the logged values.

If `-run_inference` is set to true during training inference will be ran after `-inference_interval` number of epochs, inference results are saved in storage under `./{-project_dir}/{-run_name}_{TIMESTAMP}`. `-test_data_path` can contain unprocessed `.pdf`,`.jpg` (also `.jpeg`) and `.png` images.

The model used is [UNet 3+](https://arxiv.org/abs/2004.08790) with ResBlocks.

![UNet 3+ architecture](/images/unet3p_architecture.png "UNet 3+ architecture")

The same can be done using the alternative Jupyter Notebook `train.ipynb`, by default it has parameters that require less computational power and all data paths are relative to the project directory.

## Inference
To perform inference run `inference.py`.

Make sure to set up `tile_size`, `model_path`, `input_dir` and `output_dir`. By default these are relative to the project directory and should not require changes. The `model_path` is set to the best model according to evaluation IoU, `tile_size` is set to 512 (512x512), `input_dir` includes both subdirectories of `data/test_data` and `output_dir` is set to `inference_results`.

However, be warned that running inference with default arguments will overwrite the existing `inference_results` directory.

An alternative Jupyter Notebook `inference.ipynb` is available too that is identical to `inference.py`. 
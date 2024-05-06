## AI Wave Height Model
[![License](https://img.shields.io/static/v1?label=License&message=Apache&color=<Yellow>)](https://github.com/huggingface/diffusion-models-class/blob/main/LICENSE) &nbsp;
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-WeightsFile-blue?logo=HuggingFace)](https://huggingface.co/YulKeal/AI-Wave-Height-Model/tree/main)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)

AI Wave Height Model is an advanced deep learning framework that combines the power of U-Net architecture with Earthformer models to achieve accurate predictions of wave 
heights in various marine regions, including the Black Sea, Northwest Pacific, and Global Ocean.

## Introduction
This model leverages state-of-the-art deep learning techniques to forecast wave heights with high precision. By integrating the U-Net architecture, 
renowned for its excellence in image segmentation tasks, with Earthformer models tailored for spatiotemporal data analysis, 
our framework can effectively capture complex patterns and dependencies in oceanic wave dynamics.
The animation below gives the SWH prediction results of the AI model on the test set compared to the ERA5 dataset.

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/global.gif" alt="Global" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/blacksea.gif" alt="Blacksea" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/pacificocean.gif" alt="pacificocean.gif" width="800"/>
</p>

## DataSets

| DataSet Name                                                                      | Parameters Used|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)   | Sea surface wave significant height|
| [Black Sea Waves Analysis and Forecast](https://data.marine.copernicus.eu/product/BLKSEA_ANALYSISFORECAST_WAV_007_003/description) |10m u-component of wind、10m v-component of wind、Significant height of combined wind waves and swell|

## Study Region

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/f1.jpg" alt="SR" width="800"/>
</p>

| Region| Latitude and longitude range|Spatial and temporal resolution|Source of data sets| Training data vintage division (Train/Val/Test)|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|Black Sea |40.5°N - 47°N<br>27.25°E - 42°E|0.05° x 0.05°,	Hourly|Wind data from ERA5, Wave data from BSWAF|2010-2019/2022/2020|
|Northwest Pacific Ocean|Wind: <br> 55°N - 0°<br>100°E - 180°E <br> Wave: <br> 45°N - 0°<br>100°E - 135°E|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|
|Global region|70°N - 70°S<br>All longitudes|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|


## Model Weights File Definition

| Filename  | Description|spatial and temporal resolution|Input data shape|Output data shape
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|1|1|1|([batch,72,131,296],[batch,72,131,296]) |[batch,131,296]|
|1|1|1|1|1|
|1|1|1|[batch,240,281,760，2]) |[batch,1,281,720,1]|




## Credits
Third-party libraries:
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [OmegaConf](https://github.com/omry/omegaconf)
- [YACS](https://github.com/rbgirshick/yacs)
- [Pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## License

This project is licensed under the Apache-2.0 License.

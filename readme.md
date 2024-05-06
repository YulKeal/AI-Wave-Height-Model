## AI Wave Height Model
[![License](https://img.shields.io/static/v1?label=License&message=Apache&color=<Yellow>)](https://github.com/huggingface/diffusion-models-class/blob/main/LICENSE) &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-WeightsFile-blue?logo=HuggingFace)](https://huggingface.co/YulKeal/AI-Wave-Height-Model/tree/main)

AI Wave Height Model is an advanced deep learning framework that combines the power of U-Net architecture with Earthformer models to achieve accurate predictions of wave 
heights in various marine regions, including the Black Sea, Northwest Pacific, and Global Ocean，using only wind field data.

## Introduction
Drawing upon the intrinsic relationship between wind dynamics and wave generation, this research introduces an AI-driven framework tailored for predicting significant wave heights (SWH) on a global or regional scale. Given its paramount importance in oceanographic studies and the availability of high-fidelity global observations from satellite altimeters, our focus is solely directed towards SWH modeling.

The proposed model demonstrates exceptional efficiency, capable of accurately forecasting 1-year global SWH with a spatial resolution of 0.5°×0.5° and a temporal resolution of 1 hour in under 30 minutes on standard personal computers. Its remarkable speed renders it a valuable alternative to Numerical Wave Models (NWMs), particularly suitable for scenarios necessitating rapid response and judicious utilization of computational resources.

Applications of this model span a wide spectrum of domains, including but not limited to wave modeling initiatives in resource-constrained regions, ensemble forecasting endeavors during emergency situations prone to catastrophic waves, and comprehensive studies on wave climatology under varying climate scenarios.

## DataSets

| DataSet Name                                                                      | Parameters Used|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)   | Sea surface wave significant height|
| [Black Sea Waves Analysis and Forecast](https://data.marine.copernicus.eu/product/BLKSEA_ANALYSISFORECAST_WAV_007_003/description) |10m u-component of wind、10m v-component of wind、Significant height of combined wind waves and swell|

## Study Region

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/f1.jpg" alt="SR" width="700"/>
</p>

| Region| Latitude and longitude range|Spatial and temporal resolution|Source of data sets| Training data vintage division (Train/Val/Test)|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|Black Sea |40.5°N - 47°N<br>27.25°E - 42°E|0.05° x 0.05°,	Hourly|Wind data from ERA5, Wave data from BSWAF|2010-2019/2022/2020|
|Northwest Pacific Ocean|Wind: <br> 55°N - 0°<br>100°E - 180°E <br> Wave: <br> 45°N - 0°<br>100°E - 135°E|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|
|Global region|70°N - 70°S<br>All longitudes|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|

## Results

The animation below gives some of the results of a visual comparison of the AI model's predictions of SWH on the three study area test sets relying only on the wind field with the ERA5 data, and it is clear to see that the AI model's predictions are very accurate.

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/global.gif" alt="Global" width="700"/>
</p>

<div align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/blacksea.gif" alt="Blacksea" width="350" style="display: inline-block; margin-right: 20px;"/>
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/pacificocean.gif" alt="Pacificocean" width="600" style="display: inline-block;"/>
</div>


## Model Weights File Definition

| Filename  | Description|spatial and temporal resolution|Input data shape|Output data shape
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|1|1|1|([batch,72,131,296],[batch,72,131,296]) |[batch,131,296]|
|1|1|1|1|1|
|1|1|1|[batch,240,281,760，2]) |[batch,1,281,720,1]|



## License

This project is licensed under the Apache-2.0 License.

## AI Wave Height Model
[![License](https://img.shields.io/static/v1?label=License&message=Apache&color=<Yellow>)](https://github.com/huggingface/diffusion-models-class/blob/main/LICENSE) &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-WeightsFile-blue?logo=HuggingFace)](https://huggingface.co/YulKeal/AI-Wave-Height-Model/tree/main)

AI Wave Height Model is an advanced deep learning framework that combines the power of U-Net architecture with Earthformer models to achieve accurate predictions of wave 
heights in various marine regions, including the Black Sea, Northwest Pacific, and Global Ocean,using only wind field data.

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
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/f1.jpg" alt="SR" width="600"/>
</p>

| Region| Latitude and longitude range|Spatial and temporal resolution|Source of data sets| Training data vintage division (Train/Val/Test)|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|Black Sea |40.5°N - 47°N<br>27.25°E - 42°E|0.05° x 0.05°,	Hourly|Wind data from ERA5, Wave data from BSWAF|2010-2019/2022/2020|
|Northwest Pacific Ocean|Wind: <br> 55°N - 0°<br>100°E - 180°E <br> Wave: <br> 45°N - 0°<br>100°E - 135°E|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|
|Global region|70°N - 70°S<br>All longitudes|0.5° x 0.5°,	Hourly|ERA5|2000-2017/2022/2020|

## Results

The animation below gives some of the results of a visual comparison of the AI model's predictions of SWH on the three study area test sets relying only on the wind field with the ERA5 data, and it is clear to see that the AI model's predictions are very accurate.

<p align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/global.gif" alt="Global" width="600"/>
</p>

<div align="center">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/blacksea.gif" alt="Blacksea" width="270" style="display: inline-block; margin-right: 20px;"/>
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/pacificocean.gif" alt="Pacificocean" width="480" style="display: inline-block;"/>
</div>


## Model Weights File Definition



### Loading Model and Weights

| Filename  |Input data shape|Output data shape
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|black_sea_unet.pt|([batch,72,131,296],[batch,72,131,296]) |[batch,131,296]|
|pacific_ocean.pt|[batch,72,91,71,2]) |[batch,1,91,71,1]|
|pacific_ocean_extendwind.pt|[batch,72,111,161,2]) |[batch,1,91,71,1]|
|global_unet.pt|([batch,240,281,760],[batch,240,281,760]) |[batch,1,281,720,1]|
|global_ef_checkpointX.pt|[batch,240,281,760,2]) |[batch,1,281,720,1]|


The table above gives the inputs accepted by the pt file for the different regions, the pt file contains the weights and the model, you can load the model and weights directly using a code similar to the following

```python

# Load model
model = torch.load("xxx.pt")
model.eval()

# Make predictions
#For the earthformer model the UV component of the wind is placed in the last channel, so it is entered into the model as a whole
predicted_SWH = model(WIND_UV)

#For the u-net model winds the UV components are entered into the model separately, as shown in the table above
predicted_SWH = model(WIND_U,WIND_V)
```


### Bias Correction with Linear Regression
For the npz file with the suffix linear_regression, which contains the k, b values for the bias-corrected linear regression of the corresponding model, you can accomplish the correction with the following code，

```python

# Load linear regression model
lmodel = np.load("xxx_linear_regression.npz")
k = torch.reshape(torch.tensor(lmodel['k']), [1, 1, lat, lon, 1]).to(device)
b = torch.reshape(torch.tensor(lmodel['b']), [1, 1, lat, lon, 1]).to(device)

# Apply correction
predicted_SWH = predicted_SWH * k + b
```

### Removing Land and Ice Regions
For the npz file with the suffix mask in the global region are masks for land and ice regions, you can remove these regions with the following code

```python

# Load mask for land and ice regions
loaded = np.load('xxx_mask.npz')
mask = loaded['mask']
mask = np.logical_not(mask)
mask = mask[np.newaxis, :, :, np.newaxis]
mask = torch.tensor(mask).to(device)

# Apply mask to predicted_SWH
predicted_SWH = torch.masked_fill(predicted_SWH, mask, 0)
```

### Model Ensemble
The npz file with the suffix model_ensemble in the global region is the binary linear regression model used to blend the EarthFormer and Unet models, and you can implement model ensemble with the following code

```python

# Load binary regression file
lmodel=np.load("xxx_model_ensemble.npz") 
k=lmodel['k'] 
b=lmodel['b']

# Apply model ensemble
Model_Ensemble_SWH = EarthFormer * k[:,:,0] +UNet *k[:,:,1] +b
```
It is worth noting that when performing these operations, please make sure that the tensor is shaped correctly to ensure that it can be broadcast and operated on correctly. The model input and output shapes can be found in the table above.

These files are uploaded on Hugging Face and you can jump to there by clicking the button below the main title.

## License

This project is licensed under the Apache-2.0 License.

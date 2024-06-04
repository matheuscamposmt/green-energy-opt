# Waste to Energy

## Minimizando o custo de produção de biocombustíveis utilizando Otimização e Machine Learning

Inspirado no hackathon da Shell AI de 2023.

### Sobre
O projeto consiste em prever o valores de biomassa de pontos de coleta em Gujarat, na Índia.


## Forecasting
Generate Forecast
You can find the code for generating the biomass forecast in the forecasting.ipynb notebook.

Biomass Dataset Cleanup
To ensure data accuracy, the notebook addresses and fills in duplicated values that occurred before the 2014 census.

Cluster Indexes
The clustering process is primarily based on district names, followed by checking correlations for each index within each district. Each index is assigned to the district with the highest Pearson correlation.

Create Table for Crop Production
A table containing crop production data for each district is created based on Desagri data. Missing values before 2014 are filled in using production conservation ratios.

Add Elevation Map and Crop Land Map
Crop land data from EarthStat and elevation data from NASA Earth Observation NEO are integrated into the analysis.

Train Model
The model pipeline consists of a MaxAbsScaler and an ExtraTreeRegressor. Cross-validation is performed for each year based on all other years, with the following results:
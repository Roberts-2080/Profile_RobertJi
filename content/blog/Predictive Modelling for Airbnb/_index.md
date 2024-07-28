---
title: "Predictive Modelling for Airbnb: Enhancing Rental Price Predictions and Host Strategies"
summary: "Explore our project focused on developing predictive models for Airbnb rental prices and uncovering insights to help hosts optimize their listings."
date: 2022-06-01

# Featured image
# Place an image named 'featured.jpg/png' in this page's folder and customize its options here.
image:
  caption:

authors:
  - admin

tags:
  - Data Mining
  - Machine Learning
  - Predictive Modelling
---

Welcome 👋

{{< toc mobile_only=true is_open=true >}}

## Overview

In this project, we developed predictive models to estimate Airbnb rental prices and extracted valuable insights to help hosts improve their listings. By leveraging various machine learning techniques, we aim to provide actionable advice to hosts, property managers, and real estate investors.

## Project Background

Airbnb is a popular platform for short-term rentals, connecting hosts with potential renters. Our project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM), starting with business understanding, followed by data preparation, modelling, and evaluation. The goal is to build a predictive model for rental prices and discover insights that can help hosts make better decisions.

### Methodologies

- **Data Pre-processing and Cleaning**:
  - Analyzed missing values and performed data type conversions to ensure clean and usable data.

- **Exploratory Data Analysis (EDA)**:
  - Conducted univariate, bivariate, and multivariate analyses to understand data distributions and relationships.

- **Feature Engineering**:
  - Handled missing values, transformed categorical variables, and applied Box Cox and Log1p transformations to normalize data.

- **Modelling**:
  - Implemented various models, including Linear Regression, Ridge, LASSO, Tree-Based Models (LightGBM, XGBoost, Random Forests), Generalized Additive Models, and Neural Networks.
  - Evaluated model performance using metrics like RMSE, R2, and MAE.

### Experiment Setup

We used a dataset of Airbnb listings in Sydney, containing detailed information such as rental price, geolocation, property type, room type, and more. The data was pre-processed, and different machine learning models were trained and tested to predict rental prices.

### Key Results

- **Model Performance**:
  - XGBoost emerged as the best-performing model with a test RMSE of 0.4410 and R2 of 0.7546.

- **Feature Importance**:
  - Key features impacting rental prices include the number of bedrooms, accommodates, and property location.

| Model                | Test RMSE | Test R2 | Test MAE | Test RMSE on Price |
|----------------------|-----------|---------|----------|---------------------|
| Linear               | 0.4957    | 0.6899  | 0.3678   | 386.5119            |
| Ridge                | 0.4993    | 0.6854  | 0.3711   | 392.5573            |
| LASSO                | 0.4959    | 0.6897  | 0.3678   | 387.0839            |
| LightGBM             | 0.4447    | 0.7504  | 0.3227   | 345.3431            |
| XGBoost              | 0.4410    | 0.7546  | 0.3181   | 334.4902            |
| Random Forests       | 0.4618    | 0.7308  | 0.3338   | 349.2299            |
| Neural Network       | 0.5043    | 0.6791  | 0.3836   | 380.1261            |
| GAM                  | 0.4773    | 0.7126  | 0.3520   | 361.1443            |
| Stacking Model       | 0.4429    | 0.7525  | 0.3198   | 335.7857            |

## Experience Reflection

This project provided valuable insights into the factors affecting Airbnb rental prices. By using advanced machine learning techniques, we were able to build robust predictive models and uncover important features influencing rental prices. The process also highlighted the importance of thorough data pre-processing and feature engineering in developing accurate models.

### Conclusion

Our findings indicate that investing in properties with more bedrooms, higher accommodation capacity, and prime locations can maximize rental income. Hosts should also focus on improving their response time and encouraging positive reviews to enhance their listings' appeal and profitability.

[Download the full research paper](Airbnb_Price_Prediction_Report.pdf)



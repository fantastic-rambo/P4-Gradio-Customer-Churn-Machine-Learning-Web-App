# WELCOME TO MY GITHUB PAGE

![Time-Series-Forecasting-image](https://images.startups.co.uk/wp-content/uploads/2023/05/sales-forecast.jpg?width=709&height=460&fit=crop)

Elevating Retail Analytics: Predicting Ecuador's Grocery Sales - The Favorita Forecasting Challenge!



# Time Series Forecasting Project

Welcome to the exciting world of time series forecasting, where we embark on a journey to predict store sales for Favorita, one of Ecuador's largest and most prominent grocery retailers.

In this data-driven project, we delve deep into the intricate art of predictive analytics, armed with historical sales data, cutting-edge machine learning techniques, and the drive to optimize the future of retail.

## Project Overview

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to explore and analyze sales for store sales for Favorita. The aim is to leverage data-driven insights to identify models that can accurately predict the value of the dependent variable based on the values of the independent variables.

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Data Dictionary](#data-dictionary)
- [Project Highlights](#project-highlights)
- [Summary](#summary)
- [Hypothesis Investigated](#hypothesis-investigated)
  - [Results](#results)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Selection](#model-selection)
- [Recommendations](#recommendations)
- [Getting Started](#getting-started)
- [License](#license)
- [Author](#author)


## Project Structure📂

- `code/`: Contains the dataset used for analysis and the Jupyter notebook detailing the data exploration, preprocessing, and model building steps.
- `article/`: Holds project-related article.
- `LICENSE`: Project license.
- `README.md`: Project overview, links, highlights, and information.

## Data Dictionary

| **Dataset** | **Description** |
|------------|-----------------|
| train.csv  | Training data containing time series of features store_nbr, family, and onpromotion, as well as the target sales. |
|            | - `store_nbr`: Identifies the store where the products are sold. |
|            | - `family`: Identifies the type of product sold. |
|            | - `sales`: Total sales for a product family at a specific store on a given date (can be fractional). |
|            | - `onpromotion`: Total number of items in a product family that were being promoted at a store on a given date. |
| test.csv   | Test data with the same features as the training data. Predict target sales for these dates. |
| transaction.csv | Contains date, store_nbr, and transactions made on specific dates. |
| sample_submission.csv | Sample submission file in the correct format. |
| stores.csv | Store metadata, including city, state, type, and cluster. |
|            | - `cluster`: Grouping of similar stores. |
| oil.csv    | Daily oil price data, including values during both the train and test data timeframes. |
| holidays_events.csv | Holidays and events data, with metadata. |


# Project Highlights🚀

- Employed a holistic approach, embracing the CRISP-DM framework, to gain a deep understanding of retail dynamics.
- Mined invaluable insights from extensive exploratory data analysis, unveiling hidden trends and patterns within the dataset.
- Engineered advanced predictive models, featuring the formidable XGBoost algorithm, to forecast sales with unprecedented accuracy.
- Implemented rigorous hyperparameter tuning, unlocking the full potential of our models and achieving unparalleled predictive performance.
- Crafted a compelling and informative article, sharing the project's compelling journey, groundbreaking results, and its potential to reshape the future of retail forecasting.

## Summary

| Code | Name                                     |                                             Published Article                                              |                                                                                                                                                    Deployed Dashboard |
| ---- | ---------------------------------------- | :--------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LP3  | Sales Time Series Forecasting(Prediction)| [Read Article](https://medium.com/@isaacrambo/revealing-the-power-of-time-series-forecasting-predicting-ecuadors-grocery-sales-with-precision-8c3b0bac97be) | [View Dashboard](https://app.powerbi.com/view?r=eyJrIjoiZjY1ZjkxYjktYjhkNy00OWRmLTkyM2QtZWVjMTVhMmY2ZTczIiwidCI6IjQ0ODdiNTJmLWYxMTgtNDgzMC1iNDlkLTNjMjk4Y2I3MTA3NSJ9) |

## Hypothesis Investigated

**Null Hypothesis (H0)** : The number of products under promotion does not influence sales in supermarkets.

**Alternate Hypothesis (H1)** : The number of products under promotion significantly influence sales in supermarkets.

## Rationale

The rationale for testing these hypotheses is to determine whether there is empirical evidence to support the idea that promotions have a meaningful impact on sales in supermarkets.

By testing these hypotheses and examining the correlation between promotions and sales, businesses can gain valuable insights into the dynamics of supermarket sales and make evidence-based decisions regarding their promotional strategies.


### Results

| Test Conducted               | Pearson Correlation     | P-Value                |
| ---------------------------- | ------------------ | ---------------------- |
| Independent Samples T - Test | 0.4180 | 0.0000 |

In conclusion, the Pearson correlation coefficient calculated between the number of products under promotion (as indicated by the "onpromotion" column) and sales in supermarkets is approximately 0.4180. The corresponding p-value obtained from the correlation analysis is very close to zero (P-value: 0.0000). Based on the results of this analysis, we reject the null hypothesis.

There is a statistically significant positive correlation (Pearson Correlation Coefficient = 0.4180) between the number of products under promotion and sales in supermarkets. This suggests that promotions have a significant influence on sales, and as the number of products under promotion increases, sales tend to increase as well.



## Exploratory Data Analysis (EDA)📊


A snapshot of the conducted exploratory data analysis, aimed at addressing pivotal business inquiries during the analysis process.

| ![storesbytype](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/dae0298b-2477-4772-a650-31a74b839266)         | ![storesbystate](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/549cf603-45e6-43a6-9304-760c36c5f324)       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![oil trend](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/468f4fe4-84d4-4e7a-a58a-d85739646a49) | ![newplot](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/79b272b7-319c-48de-901c-467ca57b9a78) |

## Model Selection


![modelse](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/bb9e2da1-46cf-4dd2-91a7-94acf7f070ff)
After carefully assessing the performance of our models using key evaluation metrics, it is evident that the XGBoost model stands out as the most effective choice for our dataset. The RMSLE (Root Mean Squared Logarithmic Error) serves as a crucial indicator, and the XGBoost model achieved the lowest RMSLE of 0.0054 among all models evaluated. This indicates that the XGBoost model provides the most accurate and precise predictions when compared to ARIMA, SARIMA, and ETS models.

Therefore, for this specific forecasting task, we are adopting the XGBoost model for its superior predictive accuracy.


## Recommendations

1. **Promotion Optimization:** Based on the analysis of the impact of promotions on sales, consider optimizing promotion strategies. Identify which types of promotions (e.g., discounts, BOGO offers) have the most significant influence on sales and tailor promotional campaigns accordingly. By focusing promotional efforts on what truly drives sales, you can maximize the return on investment.

2. **Focus on High-Performing Cities**: The top-performing city, "Quito," stands out with the highest sales. It's essential to allocate additional resources and marketing efforts to maintain and potentially increase sales in Quito. Additionally, cities like "Guayaquil," "Cuenca," "Ambato," and "Santo Domingo" have also shown strong sales performance. Consider developing city-specific strategies to capitalize on these markets.

3. **Cluster-Centric Approach**: The analysis reveals that certain clusters, such as "Cluster 14," "Cluster 6," and "Cluster 8," exhibit remarkable sales figures. Invest in understanding the unique characteristics of these clusters and tailor product assortments, promotions, and inventory management strategies to maximize sales potential in these areas.

4. **Cross-Analysis Opportunities**: Explore opportunities to combine the strengths of high-performing cities, clusters, store types, and states. For example, consider aligning promotions with holidays and events in top cities and clusters to maximize sales impact. Additionally, assess whether specific store types thrive in particular cities or clusters, and use this information to refine expansion plans.

## Getting Started🏁

1. Clone this repository: `git clone https://github.com/fantastic-rambo/LP3-Super-Store-Sales-Forecasting-ML.git`
2. Navigate to the project directory: `LP3-Super-Store-Time-Series-Forecasting`
3. Explore the Jupyter notebooks for detailed steps and code execution.
4. Read the published article for a comprehensive understanding of the project.

## License📜

This project is licensed under the [MIT License](LICENSE).

## Author✍️

Isaac Agbogah Rambo

Connect with me on LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/isaac-agbogah/)

---

Feel free to star ⭐ this repository if you find it helpful!

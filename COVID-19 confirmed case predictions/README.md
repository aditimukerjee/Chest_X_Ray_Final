# ENEL645 A Deep Learning Framework for COVID-19 Outbreak Prediction
## Introduction
At Deep ImageJ we are fighting against COVID-19 by first visualizing the spread of this pandemic in real-time to understand which countries worldwide have the highest level of confirmed cases and deaths due to COVID-19. We then intend to deploy the deep learning LSTM model that will predict the rise in the number of confirmed COVID-19 cases expected in Canada in the future. The objective behind building and deploying this highly reliable deep learning model is that its predictions will help the Canadian government take the necessary lockdown measures to stop the spread of COVID-19 to flatten the curve.

## Methodology
Dataset for prediction: The publicly available John Hopkin University’s (JHU’s) COVID-19 dataset was used to develop a dashboard depicting how COVID-19 is affecting countries worldwide[5]. This JHU dataset were utilized to deploy the LSTM model which was used to predict the number of confirmed COVID-19 cases in Canada. To make our analysis easier from the JHU dataset, the data for only the last three months, January 1, 2021 to April 9, 2021 was extracted. Model will predict daily new confirmed cases, and predicted cases between March 31, 2021 to April 9, 2021 will be compared with actual value. In the report whenever we have mentioned till date, it means till March 4, 2021 the last date the data was extracted.

## Data Visualization 
To first understand the magnitude of the problem at hand, JHU’s dataset was used to develop a dashboard that illustrates the total number of confirmed COVID-19 cases and deaths worldwide to date. Bootstrap, Plotly, and Django were utilized to create this dashboard[5][15].

## Model Building And Deployment
#### Data Preprocessing
This extracted dataset was then split into training and test as well as valid datasets with a ratio of 6:2:2. 
●	Imputation - The dataset did not contain missing values so no data imputation was applied. 
●	Normalization - Min-Max scaling was utilized to normalize the data since the LSTM is very sensitive to the input data. Data normalization was performed for the training dataset only. 
#### Model deployed
Firstly, LSTM model was deployed to address this supervised regression problem[8][9][10][11][12]. The LSTM model deployed was a sequential model consisting of 5 fully connected layers, followed by 3 LSTM layers, 3 dropout layers, 1 dense output layer. The LSTM model consisted of approximately 50853 parameters and was relatively easy to train and test. Secondly, GRU model was deployed to address this. The GRU model deployed was a sequential model consisting of 5 fully connected layers, followed by 3 GRU layers, 1 dense output layer. Thirdly, Bidirectional RNN model consisted of approximately 24071 parameters, consisting of one GRU layer and tow bidirectional GRU layers.


#### Building the model
For the Bidirectional RNN model the following hyperparameters and metrics were employed: 
●	Adam was used as an optimizer with a learning rate of 0.0001
●	Tanh and Sigmoid functions was used for every gate.
●	Being a supervised regression problem, the system performance was measured using two metrics namely, the mean squared error (MSE) and R-squared. 
●	Number of epochs of 600 and a batch size of 10 were considered for training the model.
●	No callbacks were defined.

## Results and Discussion

#### Model

![image](https://user-images.githubusercontent.com/77630658/114956838-47af0b80-9e92-11eb-8192-f68cad980418.png)

Note 1: The actual active cases on the tenth day is fixed as 0.
Figure 2: Performance of the Bidirectional RNN, LSTM, and GRU models in predicting the confirmed COVID-19 cases in Canada for 9 days in a row 
The performance of all the three models, Bidirectional RNN, LSTM, and GRU models in predicting the number of confirmed COVID-19 cases in Canada from March 31, 2021 to April 9, 2021 i.e. 9 days in row is depicted above in Figure 2. As shown above, in Figure 2, the Bidirectional RNN model with an MSE and R-squared of 0.0689 and -0.9339 respectively, performs much better than the other two models, LSTM, and GRU.
 
The actual number of confirmed COVID-19 cases is very close to the predicted COVID-19 for day 1, and day 3 for all three models. After day 3 the predictions are very different from the actual confirmed COVID-19 cases. 
 

 
![image](https://user-images.githubusercontent.com/77630658/114956887-644b4380-9e92-11eb-8072-f51b85a40327.png)

Note 1: The actual active cases on the tenth day is fixed as 0.
Figure 3: Performance of Bidirectional RNN model in predicting the confirmed COVID-19 cases in UK (top left) , Italy, Japan and Israel (bottom right) for 9 days in a row 
 
As the bidirectional RNN model was the best performing model while making predictions for the confirmed COVID-19 cases for Canada, we used the same fine-tuned model to predict the number of confirmed COVID-19 cases for four other countries, namely UK, Italy, Japan and Israel. 
 
As shown above, in Figure 3, the Bidirectional RNN model with an MSE and R-squared of 0.0159 and 0.041 respectively, is able to predict the trend in the confirmed COVID-19 cases in Italy, and UK remarkably well. The predicted confirmed COVID-19 cases for both these countries are very close to the actual cases. However the prediction of the confirmed COVID-19 cases vary a lot more from actual confirmed COVID-19 cases for both Israel and Japan. The MSE and R-squared values for both these countries are on a higher side as compared to Italy, and UK.

Video link to the presentation: https://youtu.be/NUc34Dm0mK4





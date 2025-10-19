import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error

housing_data=pd.read_csv('House_Pricing.csv')
numerical_data = housing_data.select_dtypes(include=['int64','float64'])
categorical_data = housing_data.select_dtypes(include=['object'])


housing_data= housing_data.dropna(subset='Sale Price')
housing_data.loc[:,'No of Bathrooms'] = housing_data['No of Bathrooms'].fillna(housing_data['No of Bathrooms'].median())
housing_data.loc[:,'Flat Area (in Sqft)'] = housing_data['Flat Area (in Sqft)'].fillna(housing_data['Area of the House from Basement (in Sqft)'] + housing_data['Basement Area (in Sqft)'])
housing_data.loc[:,'Area of the House from Basement (in Sqft)']=housing_data['Area of the House from Basement (in Sqft)'].fillna(housing_data['Flat Area (in Sqft)']-housing_data['Lot Area (in Sqft)'])
housing_data.loc[:,'Lot Area (in Sqft)']=housing_data['Lot Area (in Sqft)'].fillna(housing_data['Lot Area after Renovation (in Sqft)'])
housing_data=housing_data.drop(['No of Times Visited','Zipcode','Latitude', 'Longitude','Lot Area (in Sqft)','Renovated Year','ID','Area of the House from Basement (in Sqft)','Basement Area (in Sqft)','Date House was Sold','Flat Area (in Sqft)','Zipcode'],axis=1)
housing_data = housing_data.dropna(subset=['Living Area after Renovation (in Sqft)'])


#ordinal encoding
ordinal_enc=OrdinalEncoder(categories=[['Bad','Okay','Fair','Good','Excellent']])
housing_data['Condition of the House']=ordinal_enc.fit_transform(housing_data[['Condition of the House']])

ordinal_encoder=OrdinalEncoder(categories=[['No','Yes']])
housing_data['Waterfront View']=ordinal_encoder.fit_transform(housing_data[['Waterfront View']])

y=housing_data['Sale Price']
x=housing_data.drop(columns='Sale Price')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

# Select top k features using f_regression
selector = SelectKBest(score_func=f_regression, k='all') # You can adjust k as needed
selector.fit(x_train, y_train)

# Get the scores for each feature
feature_scores = pd.DataFrame({'Feature': x_train.columns, 'Score': selector.scores_})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Select the top features based on the scores (e.g., top 5)
selected_features = feature_scores['Feature'][:5].tolist()
print("Selected features:", selected_features)

# Create new dataframes with only the selected features
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Initialize and train the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(x_train_selected, y_train)

# Make predictions on the test set
y_pred_gb = gb_model.predict(x_test_selected)

# Evaluate the Gradient Boosting model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print("\nGradient Boosting Regressor Evaluation:")
print(f"Mean Squared Error: {mse_gb}")
print(f"Root Mean Squared Error: {rmse_gb}")
print(f"R-squared: {r2_gb}")
print(f"Mean Absolute Error: {mae_gb}")

with open("model.pkl", "wb") as f:
    pickle.dump(gb_model, f)




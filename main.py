import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib

data = pd.read_csv("instagram_accounts.csv")

dFrame = pd.DataFrame(data)

dFrame = dFrame.drop(columns=["Name","Channel Info","Category","No of Followers","No of Posts","Avg No of Likes"])

activity_threshold = 730
daily_hrs_threshold = 120

# Find the minimun time since last action for each row
dFrame['Min_Time_Since_Action'] = dFrame[['Time since last post', 'Time since last Like', 'Time since last Comment']].min(axis=1)

# Add churned column based on conditions
def assign_churned(row):
  min_time = row['Min_Time_Since_Action']
  avg_daily_time = row['Avg_daily_time_spend']
  if min_time < activity_threshold:
    return 0
  elif avg_daily_time > daily_hrs_threshold:
    return 0
  else:
    return 1

dFrame['Churned'] = dFrame.apply(assign_churned, axis=1)

# Split features and target variable
X = dFrame[['Time since last post','Time since last Like','Time since last Comment','Avg_daily_time_spend']]
y = dFrame['Churned']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Print the accuracy of the model
# print("Accuracy:", model.score(X_test, y_test))




# Load your pre-trained Logistic Regression model (assuming it's saved as 'model.pkl')
# newmodel = LogisticRegression.loa
# newmodel = LogisticRegression()

# Function to predict churn based on user input
def predict_churn(time_since_post, time_since_like, time_since_comment, avg_daily_time):
  user_data = pd.DataFrame({
      'Time since last post': [time_since_post],
      'Time since last Like': [time_since_like],
      'Time since last Comment': [time_since_comment],
      'Avg_daily_time_spend': [avg_daily_time]
  })
  prediction = model.predict(user_data)[0]
  return prediction

# Streamlit app layout
st.title("Churn Prediction App")

# Input fields for user data
time_since_post = st.number_input("Time Since Last Post (Days)", min_value=0)
time_since_like = st.number_input("Time Since Last Like (Days)", min_value=0)
time_since_comment = st.number_input("Time Since Last Comment (Days)", min_value=0)
avg_daily_time = st.number_input("Average Daily Time Spent (Minutes)", min_value=0)

# Predict button
if st.button("Predict Churn"):
  prediction = predict_churn(time_since_post, time_since_like, time_since_comment, avg_daily_time)
  if prediction == 0:
    st.success("Not Churned (Likely to Stay)")
  else:
    st.warning("Churned (Likely to Leave)")
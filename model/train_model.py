import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# z=w1​x1​+w2​x2​+w3​x3​+w4​x4​+b σ(z)=1/1+e^−z  give output between 0 to 1
# if output >=0.5 -> 1 , output<0.5 -> 0 
from sklearn.metrics import  accuracy_score

data = pd.read_csv("data\student_data.csv")

x=data[['study_hours','attendance','previous_score','assignments_completed']]

y=data['result']

# split the data for train and test 80% fro training and 20% fro testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=55)

# creating a model
model=LogisticRegression()

# train the model
model.fit(x_train,y_train)

# predict
y_predict = model.predict(x_test)

# accuracy =No of correct predictions/total No of predictions
accuracy = accuracy_score(y_predict,y_test)

# accuracy of the model
print("Total accuracy",accuracy)

# Take input from the user
try:
  print("Please Enter the details to predict Pass/Fail")

  study_hours = int(input("Enter Number of Study hours :"))
  attendance= int(input("Attendance %: "))
  previous_score = int(input("Previous score: "))
  assignments_completed  = int(input("Assignments completed: "))

  # convert user input into dataframe
  new_student=pd.DataFrame({
    'study_hours':[study_hours],
    'attendance':[attendance],
    'previous_score':[previous_score] ,
    'assignments_completed':[assignments_completed] 
  })
  new_data_prediction=model.predict(new_student)
  new_student['result']=new_data_prediction[0]
  # printing the prediction
  if(new_data_prediction[0]==0):
    print("Student will Fail")
  else:
    print("Student will Pass")

  is_duplicate = (
    (data['study_hours'] == study_hours) &
    (data['attendance'] == attendance) &
    (data['previous_score'] == previous_score) &
    (data['assignments_completed'] == assignments_completed) &
    (data['result'] == new_data_prediction[0])
  ).any()
  if is_duplicate:
    print("⚠️ Record already exists. Not saved.")
  else:
    new_student.to_csv(
      "data\student_data.csv",
      mode='a',
      header=False,
      index=False
    )
    print("✅ New record saved successfully.")
except:
  print("please Enter valid data")



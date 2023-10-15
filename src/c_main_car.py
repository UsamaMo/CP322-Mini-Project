from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features.to_numpy()
y = car_evaluation.data.targets.to_numpy()
  
print("X array: ", X)
print("Y array: ", y)

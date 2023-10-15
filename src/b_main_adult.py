from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  

x_array = X.to_numpy()
y_array = y.to_numpy()



print(x_array)
print(y_array)


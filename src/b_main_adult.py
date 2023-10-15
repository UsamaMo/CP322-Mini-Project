from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features.to_numpy()
y = adult.data.targets.to_numpy()
  

print("X array: ", X)
print("Y array: ", y)

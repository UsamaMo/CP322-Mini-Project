from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

x_array = X.to_numpy()
y_array = y.to_numpy()



print(x_array)
print(y_array)


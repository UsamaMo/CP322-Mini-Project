from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features.to_numpy() 
y = heart_disease.data.targets.to_numpy()

print("X array: ", X)
print("Y array: ", y)

import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

root = Path(__file__).parent
data_path = root/"data"/"raw"/"dummy_students.csv"
data = pd.read_csv(data_path)

train,test = train_test_split(data,test_size=0.20,random_state=42)
print(train.shape)
print(test.shape)

train_path = root/"data"/"clean"/"train.csv"
#train_path.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(train_path),exist_ok=True)
train.to_csv(train_path, index=False)

test_path = root/"data"/"clean"/"test.csv"
#test_path.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(test_path),exist_ok=True)
test.to_csv(test_path, index=False)






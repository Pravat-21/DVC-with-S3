import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
from sklearn import set_config
set_config(transform_output='pandas')

scale = StandardScaler()

#root 
root = Path(__file__).parent
test_path = root/"data"/"clean"/"test.csv"
train_path = root/"data"/"clean"/"train.csv"

train_df =pd.read_csv(train_path, usecols=['eng','math','bengali','cgpa','package'])
test_df =pd.read_csv(test_path,usecols=['eng','math','bengali','cgpa','package'])

train_sc_df = scale.fit_transform(train_df)
test_sc_df = scale.transform(test_df)

train_sc_path = root/"data"/"scale"/"train_sc.csv"
os.makedirs(os.path.dirname(train_sc_path),exist_ok=True)
train_sc_df.to_csv(train_sc_path, index = False)

test_sc_path = root/"data"/"scale"/"test_sc.csv"
os.makedirs(os.path.dirname(test_sc_path),exist_ok=True)
test_sc_df.to_csv(test_sc_path, index = False)


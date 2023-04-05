import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from utils import seed_everything
import matplotlib.pyplot as plt

# seed and set paths
seed_everything(871)
train_path = '/root/ret/labelssubset.csv'
repo_path =  './' # fill this in

# load all data from train.csv
df_all = pd.read_csv(train_path)

# Cast outcome cols to strings and concatenate them to create one outcome
y_cols = ['patient_sex']

# for col in y_cols:
#     df_all[col] = df_all[col].astype(str)

# df_all['y_str'] = df_all[y_cols].agg('-'.join, axis=1)

# encode each unique set of outcomes as a unique int
enc = LabelEncoder()
df_all['y'] = enc.fit_transform(df_all['patient_sex'])
print(df_all.head(5))
#plt.hist(df_all.y)
#plt.savefig('y_hist.png')

# Create an 80/20 train-val/test split
# splits should not share PatientIDs and preserve label frequencies
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True)
for train, test in sgkf.split(df_all['image_id'], df_all['y'], groups=df_all['patient_id']):
    train_ids = df_all['image_id'][train]
    test_ids = df_all['image_id'][test]


# save study instance ids into train.txt and test.txt files
study_ids_train = list(train_ids.values)
study_ids_train = [str(i) + '\n' for i in study_ids_train]

study_ids_test = list(test_ids.values)
study_ids_test = [str(i) + '\n' for i in study_ids_test]

with open(repo_path + "train.txt", "w") as f_train:
    f_train.writelines(study_ids_train)

with open(repo_path + "test.txt", "w") as f_test:
    f_test.writelines(study_ids_test)
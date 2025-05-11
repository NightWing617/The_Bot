
# encode_variables.py

from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df, categorical_cols):
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

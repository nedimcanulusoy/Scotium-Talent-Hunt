import warnings
import matplotlib
import pandas as pd
import yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from external_helpers import dataframe_summary

matplotlib.use('agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def pre_processed_data():
    with open("paths.yaml", "r") as f:
        paths = yaml.safe_load(f)

    df_att = paths["datasets"]["attributes"]
    df_plab = paths["datasets"]["potential_labels"]

    df_att = pd.read_csv(df_att, sep=';')
    df_plab = pd.read_csv(df_plab, sep=';')

    df_att.columns = [col.upper() for col in df_att.columns]
    df_plab.columns = [col.upper() for col in df_plab.columns]

    # merge datasets on specific columns
    cols_to_merge = [col.upper() for col in ["task_response_id", "match_id", "evaluator_id", "player_id"]]
    df = pd.merge(df_att, df_plab, on=cols_to_merge)

    # Save merged dataset
    df.to_csv(paths["datasets"]["merged"], sep=',', index=False)

    cat_cols, num_cols, cat_but_car = dataframe_summary(dataframe=df)

    # drop goalkeeper attributes from POSITION_ID
    df = df[df["POSITION_ID"] != 1]

    # drop below_average from POTENTIAL_LABEL
    df = df[df["POTENTIAL_LABEL"] != "below_average"]

    # create pivot_table where index is player id, position id and potential label, columns are attribute id and
    # values are attribute value. Then reset index and convert the names of the "attribute_id" columns to string
    dff = df.pivot_table(index=["PLAYER_ID", "POSITION_ID", "POTENTIAL_LABEL"], columns="ATTRIBUTE_ID",
                         values="ATTRIBUTE_VALUE").reset_index().rename_axis(None, axis=1)
    dff.columns = dff.columns.astype(str)

    cat_cols_pvt, num_cols_pvt, cat_but_car_pvt = dataframe_summary(dataframe=dff)

    # Label encoder
    label_encoder = LabelEncoder()
    dff["POTENTIAL_LABEL"] = label_encoder.fit_transform(dff["POTENTIAL_LABEL"])

    # Apply standart scaler to numerical columns
    standart_scaler = StandardScaler()
    dff[num_cols_pvt] = standart_scaler.fit_transform(dff[num_cols_pvt])

    # Save preprocessed dataset
    dff.to_csv(paths["datasets"]["preprocessed"], sep=',', index=False)

    return dff

# IMPORTS


import pandas as pd
import pycaret.classification as clf
from path import Path

from transformer.dbconnector import DBConnector

# 1.0 READ DATA

# Connect to SQL Database

db_connector = DBConnector()

# Subscribers

subscribers_df = db_connector.read_data_from_db(table_name="Subscribers")

# Transitions


transition_df = db_connector.read_data_from_db(table_name="Transactions")


# 2.0 SIMPLIFIED DATA PREP
# Getting all the subscribers that made purchase.
subscribers_joined_df = subscribers_df.assign(
    made_purchase=lambda x: x["user_email"]
    .isin(transition_df["user_email"].unique())
    .astype(int)
)


# 3.0 QUICKSTART MACHINE LEARNING WITH PYCARET

df = subscribers_joined_df[["member_rating", "country_code", "made_purchase"]]


# * Subset the data ----

clf1 = clf.setup(
    data=df, target="made_purchase", train_size=0.8, session_id=123
)

# * Make A Machine Learning Model ----

xgb_model = clf.create_model(
    estimator="xgboost",
)


# * Finalize the model ----

xgb_finalised_model = clf.finalize_model(xgb_model)

# * Predict -----

new_df = pd.DataFrame(
    dict(
        member_rating=[4],
        country_code=["us"],
    )
)


clf.predict_model(estimator=xgb_finalised_model, data=new_df, raw_score=True)

if ~Path("models").exists():
    Path("models").mkdir()

# Save the model
clf.save_model(
    model=xgb_finalised_model, model_name="models/js_xgb_model_finalized"
)


# load the model

clf.load_model(model_name="models/js_xgb_model_finalized")


# KEY QUESTIONS:
# * SHOULD WE EVEN TAKE ON THIS PROJECT? (COST/BENEFIT)
# * MACHINE LEARNING MODEL - IS IT GOOD?
# * WHAT CAN WE DO TO IMPROVE THE MODEL?
# * WHAT ARE THE KEY FEATURES IN THE MODEL?
# * CAN WE EXPLAIN WHY CUSTOMERS ARE BUYING / NOT BUYING?
# * CAN THE COMPANY MAKE A RETURN ON INVESTMENT FROM THIS MODEL?

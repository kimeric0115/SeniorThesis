import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order
import os

folder_path = r"C:\Users\user\Documents\Senior Thesis Excel Files"
file_path = r"C:\Users\user\Documents\Senior Thesis Excel Files\output3.xlsx"

category_names = ["Comprehensive Housing Index", "Apartment", "Low-Rise Multi-Unit Housing", "Single-Family House"]
# column_names = ["Date",
#                 "Combined Jeonse and Monthly Rent Index",
#                 "Integrated Monthly Rent Price Index",
#                 "Monthly Rent Supply and Demand Trends",
#                 "Property Price Index",
#                 "Price/Jeonse",
#                 "Ratio of Monthly Rent Deposit to Jeonse Price",
#                 "Price/Nonownership Ratio",
#                 "Price/Rent",
#                 "Component Non-Ownership",
#                 "Component Jeonse",
#                 "Component Rent",
#                 "Component Price",
#                 "Policy Dummy"]
df_arr = []

for i in range(len(category_names)):
    df_arr.append(pd.read_excel(file_path, sheet_name=category_names[i]))

models = {}
type = ["Combined", "Rent", "Jeonse", "Price"]

for temp in type:
    models = {}
    i = 0

    # OLS Regression
    for df in df_arr:
        # Prepping the data by making the indices readable/sortable for Pandas
        df["Date"] = pd.to_datetime(df["Date"].astype(str),format="%Y.%m")
        df = df.sort_values("Date").set_index("Date")

        event_dates = df.index[df["Policy Dummy"] ==1]

        month_index = df.index.year * 12 + df.index.month
        event_month_index = event_dates.year * 12 + event_dates.month

        # y_combined, _ = sm.tsa.filters.hpfilter(df["Price/Nonownership Ratio"],lamb = 1600)
        # y_rent, _ = sm.tsa.filters.hpfilter(df["Price/Rent"],lamb = 1600)
        # y_jeonse, _ = sm.tsa.filters.hpfilter(df["Price/Jeonse"], lamb = 1600)

        y_price = df["Component Price"]
        y_combined = df["Component Non-Ownership"]
        y_rent = df["Component Rent"]
        y_jeonse = df["Component Jeonse"]

        if temp=="Combined":
            df["y"] = y_combined
        elif temp == "Rent":
            df["y"] = y_rent
        elif temp == "Price":
            df["y"] = y_price
        else:
            df["y"] = y_jeonse

        # Window Stacking: Building df_es by concatenating several event windows together
        windows = []
        
        for ev in event_dates:
            ev_month = ev.year * 12 + ev.month

            df_temp = df.copy()
            df_temp["event_time"] = month_index - ev_month
            df_temp = df_temp[(df_temp["event_time"] >= -12) & (df_temp["event_time"] <= 12)].copy()

            df_temp["event_id"] = ev
            windows.append(df_temp)
        
        df_es = pd.concat(windows).sort_index()

        # Create dummy variables for each event horizon
        event_cols = []
        for h in range(-6,13):
            if h == -1: # we want to use the period before the event as a baseline
                continue
            colname = f"event_{h}"
            df_es[colname] = (df_es["event_time"] == h).astype(int)
            event_cols.append(colname)

        df_es["ratio_lag1"] = df_es.groupby("event_id")["y"].shift(1)
        df_es["ratio_lag2"] = df_es.groupby("event_id")["y"].shift(2)


        lag_cols = ["ratio_lag1", "ratio_lag2"]

        tmp = df_es[event_cols + lag_cols].dropna()
        event_cols = [c for c in event_cols if tmp[c].nunique() > 1]

        y = df_es["y"]
        X = df_es[event_cols + lag_cols]
        X = sm.add_constant(X)

        model = sm.OLS(y, X, missing="drop").fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 12}
        )

        #print(model.summary())

        models[category_names[i]] = model
        i += 1

    out_path = os.path.join(folder_path, f"{temp}_component_results.xlsx")  # <<< goes into same folder
    with pd.ExcelWriter(out_path) as writer:
        for name, res in models.items():
            coef_table = res.summary2().tables[1]  # coeffs, std err, t/z, p, CI
            coef_table.to_excel(writer, sheet_name=name)
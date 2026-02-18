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
#                 "Policy Dummy"]
df_arr = []

for i in range(len(category_names)):
    df_arr.append(pd.read_excel(file_path, sheet_name=category_names[i]))

models = {}
type = ["Combined", "Rent", "Jeonse"]

for temp in type:
    models = {}
    i = 0

    # OLS Regression
    for df in df_arr:
        # Prepping the data by making the indices readable/sortable for Pandas
        df["Date"] = pd.to_datetime(df["Date"].astype(str),format="%Y.%m")
        df = df.sort_values("Date").set_index("Date")

        event_dates = df.index[df["Policy Dummy"] != 0]
        event_signs = df.loc[event_dates, "Policy Dummy"].astype(int) # +1 tightening, -1 loosening

        month_index = df.index.year * 12 + df.index.month
        event_month_index = event_dates.year * 12 + event_dates.month
        
        diffs = month_index.values[:,None] - event_month_index.values # shape: (n_dates, n_events)

        # index (column) of closest event for each date
        closest_j = np.abs(diffs).argmin(axis = 1)
        
        # For each date, find the event with the smallest abs difference
        closest_diff = diffs[np.arange(diffs.shape[0]), closest_j]

        df["event_time"] = closest_diff

        # sign (+1/-1) of the closest event for each date
        df["event_sign"] = event_signs.iloc[closest_j].values
        
        df_es = df[(df["event_time"] >= -12)&(df["event_time"] <= 12)].copy()

        # Create signed dummy variables for each event horizon
        event_cols = []
        for h in range(-9,10):
            if h == -1: # we want to use the period before the event as a baseline
                continue
            colname = f"event_{h}"
            df_es[colname] = (df_es["event_time"] == h).astype(int) * df_es["event_sign"]
            event_cols.append(colname)
        
        y_combined, trend =sm.tsa.filters.hpfilter(df_es["Price/Nonownership Ratio"],lamb = 1600)
        y_rent, trend = sm.tsa.filters.hpfilter(df_es["Price/Rent"],lamb = 1600)
        y_jeonse, trend = sm.tsa.filters.hpfilter(df_es["Price/Jeonse"], lamb = 1600)

        if temp=="Combined":
            y = y_combined
            df_es["ratio_lag1"] = y_combined.shift(1)
            df_es["ratio_lag2"] = y_combined.shift(2)
        elif temp == "Rent":
            y = y_rent
            df_es["ratio_lag1"] = y_rent.shift(1)
            df_es["ratio_lag2"] = y_rent.shift(2)
        else:
            y = y_jeonse
            df_es["ratio_lag1"] = y_jeonse.shift(1)
            df_es["ratio_lag2"] = y_jeonse.shift(2)


        lag_cols = ["ratio_lag1", "ratio_lag2"]

        X = df_es[event_cols + lag_cols]
        X = sm.add_constant(X)

        model = sm.OLS(y, X, missing="drop").fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 12}
        )

        #print(model.summary())

        models[category_names[i]] = model
        i += 1

    out_path = os.path.join(folder_path, f"{temp}_results3.xlsx")  # <<< goes into same folder
    with pd.ExcelWriter(out_path) as writer:
        for name, res in models.items():
            coef_table = res.summary2().tables[1]  # coeffs, std err, t/z, p, CI
            coef_table.to_excel(writer, sheet_name=name)
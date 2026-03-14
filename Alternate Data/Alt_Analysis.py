import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order
import os

folder_path = r"C:\Users\user\Documents\Senior Thesis Excel Files\Alternate Data"
file_path = r"C:\Users\user\Documents\Senior Thesis Excel Files\Alternate Data\master_panel.xlsx"

category_names = ["Comprehensive Housing Index", "Apartment", "Low-Rise Multi-Unit Housing", "Single-Family House"]
# column_names = ["Date",
#                 "HousingType",
#                 "Region",
#                 "Combined Jeonse and Monthly Rent Index",
#                 "Integrated Monthly Rent Price Index",
#                 "Jeonse Price Index",
#                 "Property Price Index",
#                 "Price/Jeonse",
#                 "Component Non-Ownership",
#                 "Component Jeonse",
#                 "Component Rent",
#                 "Component Price",
#                 "Price/Nonownership Ratio",
#                 "Price/Rent",
#                 "Policy Dummy"]
ratio_types = ["Combined", "Rent", "Jeonse"]

panel_df = pd.read_excel(file_path)

models_by_ratio = {}

for temp in ratio_types:
    models = {}

    # run one model per housing type
    for category in category_names:
        df = panel_df[panel_df["HousingType"] == category].copy()

        # sort panel properly
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Region", "Date"]).reset_index(drop=True)

        # choose dependent variable
        if temp == "Combined":
            df["y"] = df["Price/Nonownership Ratio"]
        elif temp == "Rent":
            df["y"] = df["Price/Rent"]
        else:
            df["y"] = df["Price/Jeonse"]

        # event dates are common Seoul-wide policy dates
        event_dates = sorted(df.loc[df["Policy Dummy"] == 1, "Date"].drop_duplicates())

        # stack windows for each event and each region
        windows = []

        for ev in event_dates:
            ev_month = ev.year * 12 + ev.month

            df_temp = df.copy()
            month_index = df_temp["Date"].dt.year * 12 + df_temp["Date"].dt.month
            df_temp["event_time"] = month_index - ev_month

            df_temp = df_temp[
                (df_temp["event_time"] >= -12) & (df_temp["event_time"] <= 12)
            ].copy()

            df_temp["event_id"] = ev
            windows.append(df_temp)

        if len(windows) == 0:
            print(f"No event windows found for {category} - {temp}")
            continue

        df_es = pd.concat(windows, ignore_index=True)
        df_es = df_es.sort_values(["Region", "event_id", "Date"]).reset_index(drop=True)

        # create event-time dummies
        event_cols = []
        for h in range(-6, 13):
            if h == -1:
                continue
            colname = f"event_{h}"
            df_es[colname] = (df_es["event_time"] == h).astype(int)
            event_cols.append(colname)

        # AR lags within each Region x event window
        df_es["ratio_lag1"] = df_es.groupby(["Region", "event_id"])["y"].shift(1)
        df_es["ratio_lag2"] = df_es.groupby(["Region", "event_id"])["y"].shift(2)

        lag_cols = ["ratio_lag1", "ratio_lag2"]

        # optional: region fixed effects
        region_dummies = pd.get_dummies(df_es["Region"], prefix="region", drop_first=True)

        # optional: event fixed effects
        event_dummies = pd.get_dummies(df_es["event_id"].astype(str), prefix="ev", drop_first=True)

        X = pd.concat(
            [
                df_es[event_cols + lag_cols],
                region_dummies,
                event_dummies
            ],
            axis=1
        )

        y = df_es["y"]

        # keep only rows used in regression
        reg_df = pd.concat([y, X], axis=1).dropna().copy()
        y_reg = reg_df["y"]
        X_reg = reg_df.drop(columns=["y"])

        # drop any event columns with no variation
        keep_cols = [c for c in X_reg.columns if X_reg[c].nunique() > 1]
        X_reg = X_reg[keep_cols]

        X_reg = sm.add_constant(X_reg)

        model = sm.OLS(y_reg, X_reg).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 12}
        )

        models[category] = model

    models_by_ratio[temp] = models

    # export each ratio type to its own workbook
    out_path = os.path.join(folder_path, f"{temp}_results_regions.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, res in models.items():
            coef_table = res.summary2().tables[1]
            safe_sheet = name[:31]   # Excel sheet name limit
            coef_table.to_excel(writer, sheet_name=safe_sheet)

    print(f"Saved: {out_path}")
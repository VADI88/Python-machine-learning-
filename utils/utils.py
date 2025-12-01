from functools import wraps

import pandas as pd
import pandas_flavor as pf  # type: ignore


def with_db_connection(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.engine.connect() as conn:
            return func(self, conn, *args, **kwargs)

    return wrapper


@pf.register_dataframe_method
def prepare_data(data: pd.DataFrame, id_column: str, date_column: str):
    if data is None:
        raise ValueError("No data available to prepare.")

    data = data.rename(columns={id_column: "id", date_column: "date"})

    data = data.reorder_columns(["id", "date"])  # type:ignore

    data["date"] = data["date"].dt.to_timestamp()

    return data

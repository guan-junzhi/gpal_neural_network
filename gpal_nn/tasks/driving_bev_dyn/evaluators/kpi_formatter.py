import os
# import pandas as pd
from gpal_lightning.neural_network.tasks.base.kpi_formatters.kpi_formatter import BaseKPIFormatter
from gpal_lightning.neural_network.tasks.builder import KPI_FORMATTERS

# pd.set_option('display.max_columns', 10000)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_rows', 10000)


@KPI_FORMATTERS.register_module()
class DRIVING_BEV_DYNKPIFormatter(BaseKPIFormatter):
    def format_kpi(self, kpi_dict_, dataset_identifier=None) -> str:
        # Matching the data
        return ""
       
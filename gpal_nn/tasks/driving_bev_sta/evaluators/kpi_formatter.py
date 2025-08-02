import os
# import pandas as pd
from gpal_lightning.neural_network.tasks.base.kpi_formatters.kpi_formatter import BaseKPIFormatter
from gpal_lightning.neural_network.tasks.builder import KPI_FORMATTERS

# pd.set_option('display.max_columns', 10000)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_rows', 10000)


@KPI_FORMATTERS.register_module()
class DRIVING_BEV_STAKPIFormatter(BaseKPIFormatter):
    def format_kpi(self, kpi_dict_, dataset_identifier=None) -> str:
        # Matching the data
        return ""
        # print_info = dict()
        # for class_name, class_kpi_dict in kpi_dict_.items():
        #     print_info_sub_class = dict()

        #     def dfs_print(result):
        #         for k, v in result.items():
        #             if isinstance(v, dict):
        #                 dfs_print(v)
        #             else:
        #                 print_info_sub_class[k] = v
        #     dfs_print(class_kpi_dict)
        #     print_info[class_name] = print_info_sub_class
        # data_df = pd.DataFrame(print_info)
        # print(data_df)
        # csv_file = os.path.join(self.global_config.save, f"bev_od_eval.csv")
        # data_df.to_csv(csv_file, header=None)
        # return str(data_df)

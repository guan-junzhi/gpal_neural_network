import os
import json
import rich
from rich.console import Console
from argparse import ArgumentParser

console = Console()

def read_json(json_path):
    if not os.path.exists(json_path):
        print("[ERROR] file not exists: {}".format(json_path))
        exit(1)
    else:
        with open(json_path, "rb") as f:
            json_data = json.load(f)
            return json_data

def compare_func(baseline_value, target_value, threshold, mode="greater"):
    if baseline_value == None or target_value == None:
        return None
    if baseline_value - 0. < 1e-3 or target_value - 0. < 1e-3:
        return None
    diff = (target_value - baseline_value) / baseline_value
    is_report = False
    is_good = False
    if mode == "greater":
        if diff >= threshold:
            is_report = True 
            is_good = True
        elif diff <= (0 - threshold):
            is_report = False
    else:
        if diff <= (0 - threshold):
            is_report = True
            is_good = True
        elif diff >= threshold:
            is_report = True
    result = {
        "report": is_report,
        "good": is_good,
        "diff": diff,
    }
    return result

class MetricsCompare():
    def __init__(self, baseline_path, target_path, threshold=0.1):
        self.baseline_path = baseline_path
        self.target_path = target_path
        self.threshold = threshold
    
    def set_detail_info(self, baseline_metrics, target_metrics, base_name, compare_info_dict):
        for metrics_name,  metrics in baseline_metrics.items():
            metrics_name_full = "{} | {}".format(base_name, metrics_name)
            if not isinstance(metrics, dict):
                compare_info_dict[metrics_name_full] = [metrics, target_metrics[metrics_name]]
            else:
                self.set_detail_info(metrics, target_metrics[metrics_name], metrics_name_full, compare_info_dict)
        return
    
    def abnormal_metrics_calculate(self, compare_info_dict):
        calculate_result = {
            "good":{},
            "bad": {}
        }
        for metrics_name, metrics in compare_info_dict.items():
            if "err" in metrics_name:
                mode = "less"
            else:
                mode = "greater"
            compare_result = compare_func(metrics[0], metrics[1], self.threshold, mode)
            if compare_result is not None and compare_result["report"]:
                if compare_result["good"]:
                    calculate_result["good"][metrics_name] = [metrics[0], metrics[1],compare_result["diff"]]
                else:
                    calculate_result["bad"][metrics_name] = [metrics[0], metrics[1],compare_result["diff"]]
        return calculate_result
                
    def save_and_show(self, calculate_result):
        console.print("[bold green]============GOOD JOB=============[/bold green]")
        for metrics_name, metrics in calculate_result["good"].items():
            console.print("[bold white]{}:[/bold white] {:.3f} -> [bold green]{:.3f} | {:.1f}%[/bold green]".format(metrics_name, metrics[0], metrics[1], metrics[2] * 100))
        console.print("[bold red]============NEED TO IMPROVE===========[/bold red]")
        for metrics_name, metrics in calculate_result["bad"].items():
            console.print("[bold white]{}:[/bold white] {:.3f} -> [bold red]{:.3f} | {:.1f}%[/bold red]".format(metrics_name, metrics[0], metrics[1], metrics[2] * 100))
        calculate_result["baseline_path"] = self.baseline_path
        calculate_result["target_path"] = self.target_path
        save_folder = ('/').join(self.target_path.split('/')[:-1])
        save_path = os.path.join(save_folder, "campare_result.json")
        with open(save_path, 'w') as f:
            json.dump(calculate_result, f, indent=2)
    
    def prossing(self):
        baseline_data = read_json(self.baseline_path)
        target_data = read_json(self.target_path)
        compare_info_dict = {}
        for class_name, baseline_class_metrics in baseline_data.items():
            if class_name not in target_data:
                print("[Warning] target metrics data not include {} class".format(class_name))
                continue
            target_class_metrics = target_data[class_name]
            for metrics_name, metircs in baseline_class_metrics.items():
                metrics_name_full = "{} | {}".format(class_name, metrics_name)
                if not isinstance(metircs, dict):
                    compare_info_dict[metrics_name_full] = [metircs, target_class_metrics[metrics_name]]
                else:
                    self.set_detail_info(metircs, target_class_metrics[metrics_name], metrics_name_full, compare_info_dict)
                    
        calculate_result = self.abnormal_metrics_calculate(compare_info_dict)
        self.save_and_show(calculate_result)
        

if __name__ == "__main__":
    parser = ArgumentParser(description="compare evaluation metric between two version")
    parser.add_argument("--baseline_path", type=str, default="/proc_data/guyue/model_factory/seq_low_ap_49.6/multi_class_all/BEV_OD_detail_metrics.json")
    parser.add_argument("--target_path", type=str, default="/proc_data/guyue/model_factory/seq_base49.6_data/multi_class/BEV_OD_detail_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    
    args = parser.parse_args()
    mc = MetricsCompare(args.baseline_path, args.target_path, args.threshold)
    mc.prossing()
    
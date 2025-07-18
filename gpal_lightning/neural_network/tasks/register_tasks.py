import importlib


def register_tasks(tasks_root: str, task_name: str) -> None:
    """This function is used to trigger the task register modules"""
    task_path = ".".join([tasks_root, task_name, "task"])
    print(f"task_path = {task_path}")
    importlib.import_module(task_path)

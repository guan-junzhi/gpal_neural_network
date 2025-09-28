import logging
import os
from argparse import SUPPRESS, ArgumentParser
from gpal_lightning import const


class ArgumentParserHelper:
    _parser = ArgumentParser("GpalLightning")
    _system_parser = _parser.add_argument_group(
        "system", argument_default=SUPPRESS)
    _model_parser = _parser.add_argument_group(
        "model", argument_default=SUPPRESS)
    _data_parser = _parser.add_argument_group(
        "data", argument_default=SUPPRESS)

    @classmethod
    def _parse_system(cls):
        """System related arguments"""
        cls._system_parser.add_argument(
            "--gpus", type=int, help="number of gpu cards used.")
        # TODO: not support debug mode for load data now, only print debug info log
        cls._system_parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether turn on debug mode or not, only limited images will be load on " "debug mode.",
        )

    @classmethod
    def _parse_model(cls):
        """model training/inference/evaluation related arguments"""
        cls._model_parser.add_argument(
            "--config", type=str, help="global_config.yaml. Required for training from scratch."
        )
        cls._model_parser.add_argument(
            "--load_from", type=str, help="load checkpoint from other experiment.")
        cls._model_parser.add_argument(
            "--onnx_path", type=str, help="load onnx_runtime engine from onnx file.")
        cls._model_parser.add_argument(
            "--resume_from", type=str, help="resume thcheckpoint from other experiment. --save should be empty"
        )
        cls._model_parser.add_argument(
            "--save",
            type=str,
            help="save path for model or inference results, not required for resuming " "experiment.",
        )
        cls._model_parser.add_argument(
            "--image_per_gpu", type=int, help="image batch size for each gpu card.")
        cls._model_parser.add_argument(
            "--workers_per_gpu", type=int, help="number of total workers for each gpu card.")
        cls._model_parser.add_argument(
            "--tasks", type=str, nargs="+", help="only train/evalute/inference these tasks.")
        cls._model_parser.add_argument(
            "--seed", type=int, help="fix the random seed of the job.")
        cls._model_parser.add_argument(
            "--vis", type=bool, default=False)

    @staticmethod
    def _argument_check(args):
        if "load_from" in args:
            if "config" not in args:
                args.config = f"{os.path.dirname(os.path.dirname(args.load_from))}/config.yaml"
            if "save" not in args:
                args.save = os.path.dirname(os.path.dirname(args.load_from))

        if "config" in args:
            assert args.save, "Save path is requred for model training."
        else:
            assert "resume_from" in args or "load_from" in args
        assert not ("resume_from" in args and "load_from" in args)

        if "resume_from" in args:
            if "config" not in args:
                args.config = f"{os.path.dirname(os.path.dirname(args.load_from))}/config.yaml"
            if "save" not in args:
                args.save = os.path.dirname(os.path.dirname(args.load_from))

        args.num_nodes = const.NUM_NODES

    @classmethod
    def parse(cls):
        cls._parse_system()
        cls._parse_model()

        args = cls._parser.parse_args()
        cls._argument_check(args)
        return args

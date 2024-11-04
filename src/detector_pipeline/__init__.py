from .detector_pipeline import *
from .ensemble_pipeline import *


ensemble_list = ["isolation_forest", "ocsvm"]


def get_model(model: str, detector_params: preproc.DetectorParameters):
    if model in ensemble_list:
            det_pipe = EnsemblePipeline(model, detector_params)

    return det_pipe


def get_model_detection_strategy(model: str):
    if model in ensemble_list:
        return "ensemble"

    return None

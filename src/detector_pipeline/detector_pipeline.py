import numpy as np

from src.utility import preproc
from src.utility.preproc import ModeData


class DetectorPipeline:
    def __init__(self, detector_params=None):
        self.run_params = detector_params.run_params
        self.preproc_params = None
        self.preprocessor = None
        self.detector = None
        self.postprocessor = None

    def train_preprocessor(self, mode_data: ModeData):
        raise NotImplementedError

    def train_detector(self, mode_data: ModeData):
        raise NotImplementedError

    def score_samples(self, window_sequence: np.ndarray):
        raise NotImplementedError

    def training_score(self, mode_data: ModeData):
        raise NotImplementedError






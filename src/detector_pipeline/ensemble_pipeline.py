import os

import numpy as np

from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import sys

from src.utility.preproc import ModeData, DetectorParameters

sys.path.append("../../")

from src.detector_pipeline.detector_pipeline import DetectorPipeline
from src.utility import preproc


def split_lists(input_list, splits):
    if splits > len(input_list):
        splits = 2

    sublists = []

    input_len = len(input_list)
    split_size = input_len // splits

    if input_len % splits:
        splits += 1

    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size

        if end > input_len:
            end = input_len

        sublists.append(input_list[start:end])

    return sublists


class KDEWrapper(KernelDensity):
    def score_inputs(self, x):
        y_hat = self.score_samples(x)
        y_hat = np.exp(y_hat)
        y_hat = np.nan_to_num(y_hat)

        return y_hat
        # TODO fix this
        # return y_hat


def classifier_scores(classifier, x):
    if isinstance(classifier, KDEWrapper):
        return classifier.score_inputs(x)

    elif isinstance(classifier, SVC):
        return classifier.predict_proba(x)[:, 0]

    else:
        return classifier.score_samples(x)


def split_outliers(x, contamination=0.9):
    abs_x = np.sum(abs(x), axis=1)
    sorted_idxs = np.argsort(abs_x)
    x = x[sorted_idxs]

    split_idx = int(len(x) * contamination)
    x0 = x[:split_idx]
    x1 = x[split_idx:]

    return x0, x1


def generate_synthetic_outliers(x, contamination=0.1, multiplier=3):
    abs_x = np.sum(abs(x), axis=1)
    sorted_idxs = np.argsort(abs_x)
    x = x[sorted_idxs]
    split_idx = int(len(x) * (1 - contamination))

    return x[split_idx:] * multiplier


def ensemble_scores(tmp_scores):
    probabilities = np.max(tmp_scores, axis=0)
    class_predictions = np.argmax(tmp_scores, axis=0)
    return probabilities, class_predictions


class EnsemblePipeline(DetectorPipeline):
    def __init__(self, detector, detector_params: DetectorParameters):
        self.run_params = detector_params.run_params
        self.preproc_params = detector_params.preproc_params
        self.model_params = detector_params.model_params
        self.modes = detector_params.modes

        self.preprocessor = {}
        for mode in self.modes:
            self.preprocessor[mode] = preproc.PreprocTools()

        self.detector = {}
        for mode in self.modes:
            if detector == "ocsvm":
                detector_model = OneClassSVM(**detector_params.model_params)

            elif detector == "isolation_forest":
                detector_model = IsolationForest(**detector_params.model_params)

            elif detector == "kde":
                detector_model = KDEWrapper(**detector_params.model_params)

            else:
                detector_model = OneClassSVM(**detector_params.model_params)

            self.detector[mode] = detector_model

        self.postprocessor = {}
        for mode in self.modes:
            self.postprocessor[mode] = {
                "standard_scaler": StandardScaler(),
                # "platt_scaler": LogisticRegression(class_weight="balanced"),
                # "min_max_scaler": MinMaxScaler(clip=True)
            }

    def fit_standard_scaler(self, mode, x0):
        x0_scores = classifier_scores(self.detector[mode], x0).reshape(-1, 1)
        self.postprocessor[mode]["standard_scaler"] = self.postprocessor[mode]["standard_scaler"].fit(x0_scores)

    def fit_platt_scaler(self, mode, x0, x2):
        x0_scores = classifier_scores(self.detector[mode], x0)
        x2_scores = classifier_scores(self.detector[mode], x2)

        x0_labels = np.zeros((x0_scores.shape[0]))
        x2_labels = np.ones((x2_scores.shape[0]))

        scores = np.concatenate((x0_scores, x2_scores)).reshape(-1, 1)
        scores = self.postprocessor[mode]["standard_scaler"].transform(scores)
        labels = np.concatenate((x0_labels, x2_labels))

        self.postprocessor[mode]["platt_scaler"] = self.postprocessor[mode]["platt_scaler"].fit(scores, labels)

    def fit_min_max_scaler(self, mode, x0, x2, quantile=0.5):
        x0_scores = classifier_scores(self.detector[mode], x0).reshape(-1, 1)
        x2_scores = classifier_scores(self.detector[mode], x2).reshape(-1, 1)

        x0_scores = self.postprocessor[mode]["standard_scaler"].transform(x0_scores).reshape(-1, 1)
        x2_scores = self.postprocessor[mode]["standard_scaler"].transform(x2_scores).reshape(-1, 1)

        # x0_scores = self.postprocessor[mode]["platt_scaler"].predict_proba(x0_scores)[:, 0]
        # x2_scores = self.postprocessor[mode]["platt_scaler"].predict_proba(x2_scores)[:, 0]

        min_max_data = [np.quantile(x0_scores, 1 - quantile), np.quantile(x2_scores, quantile)]
        min_max_data = np.array(min_max_data).reshape(-1, 1)

        self.postprocessor[mode]["min_max_scaler"] = self.postprocessor[mode]["min_max_scaler"].fit(min_max_data)

    def train_preprocessor(self, mode_data: ModeData):
        word_size = self.preproc_params.word_size
        word_stride = self.preproc_params.word_stride
        vocab_size = self.preproc_params.vocab_size

        for mode in mode_data.signals:
            array = mode_data.signals[mode]
            windows = mode_data.windows[mode]

            discretizer = preproc.form_discretizer(array)
            word_extractor = preproc.form_word_extractor(word_size, word_stride)
            vocabulary = preproc.form_vocab(array, discretizer, word_extractor, vocab_size)
            scaler = preproc.form_preproc_scaler(windows, discretizer, vocabulary, word_extractor)

            self.preprocessor[mode].discretizer = discretizer
            self.preprocessor[mode].word_extractor = word_extractor
            self.preprocessor[mode].vocabulary = vocabulary
            self.preprocessor[mode].feature_vector_scaler = scaler

    def train_detector(self, mode_data: ModeData):
        for mode in mode_data.windows:
            windows = mode_data.windows[mode]
            fv = preproc.make_stat_features(windows)

            if self.run_params.use_discrete:
                dfv = preproc.make_bow_features(
                    self.preprocessor[mode].discretizer,
                    self.preprocessor[mode].vocabulary,
                    self.preprocessor[mode].word_extractor,
                    windows
                )
                fv = np.concatenate((fv, dfv), axis=1)

            x = self.preprocessor[mode].feature_vector_scaler.transform(fv)
            # x0, _ = split_outliers(x)
            # x2 = generate_synthetic_outliers(x, multiplier=1)
            # self.detector[mode].fit(x0)

            self.detector[mode].fit(x)
            self.fit_standard_scaler(mode, x)
            # self.fit_platt_scaler(mode, x0, x2)
            # self.fit_min_max_scaler(mode, x0, x2)

    def score_samples(self, window_sequence):
        tmp_scores = self.pipeline_scores(window_sequence)
        probabilities, class_predictions = ensemble_scores(tmp_scores)

        return probabilities, class_predictions

    def pipeline_scores(self, window_seq: np.ndarray):
        tmp_scores = np.zeros((len(self.preprocessor), len(window_seq)))

        for i, mode in enumerate(self.preprocessor):
            normalized_scores = self.single_pipe_score(mode, window_seq)
            tmp_scores[i] = normalized_scores.reshape(-1)

        return tmp_scores

    def single_pipe_score(self, mode, window_seq: np.ndarray):
        use_discrete = self.run_params.use_discrete
        discretizer = self.preprocessor[mode].discretizer
        vocab = self.preprocessor[mode].vocabulary
        word_extractor = self.preprocessor[mode].word_extractor
        fv_scaler = self.preprocessor[mode].feature_vector_scaler

        detector = self.detector[mode]
        standard_scaler = self.postprocessor[mode]["standard_scaler"]
        # platt_scaler = self.postprocessor[mode]["platt_scaler"]
        # min_max_scaler = self.postprocessor[mode]["min_max_scaler"]

        fv = preproc.make_stat_features(window_seq)

        if use_discrete:
            dfv = preproc.make_bow_features(discretizer, vocab, word_extractor, window_seq)
            fv = np.concatenate((fv, dfv), axis=1)

        x = fv_scaler.transform(fv)
        scores = classifier_scores(detector, x).reshape(-1, 1)
        scores = standard_scaler.transform(scores).reshape(-1, 1)
        # scores = platt_scaler.predict_proba(scores)[:, 0].reshape(-1, 1)
        # scores = min_max_scaler.transform(scores).reshape(-1)

        return scores

    def training_score(self, mode_data: ModeData):
        """
        Computes worst ROC-AUC score for training data; in essence class distinguishability when only training on benign
        data. Only useful for tuning with training data.
        """
        roc_auc_list = []

        for mode in mode_data.windows:
            score_array = []
            label_array = []

            for window_mode, window_seq in mode_data.windows.items():
                normalized_scores = self.single_pipe_score(mode, window_seq)

                score_array.append(normalized_scores)
                labels = np.zeros(len(normalized_scores))
                labels = labels + 1 if mode == window_mode else labels
                label_array.append(labels)

            score_array = np.array(score_array).reshape(-1)
            label_array = np.array(label_array).reshape(-1)
            sample_weight = compute_sample_weight(class_weight='balanced', y=label_array)
            roc_auc_list.append(metrics.roc_auc_score(label_array, score_array, sample_weight=sample_weight))

        return np.min(roc_auc_list)











import json
import math
import os
import sys
from multiprocessing import Pool

import pandas as pd
import numpy as np

# from tqdm import tqdm
from scipy import signal, stats
from pyts import approximation, bag_of_words
from sklearn.preprocessing import StandardScaler


class RunParams:
    def __init__(self):
        self.states = []
        self.window_size = 0
        self.window_stride = 0
        self.future_size = 0
        self.future_stride = 0
        self.use_discrete = False
        self.train_new = False


class PreprocParams:
    def __init__(self, parameters: dict, use_discrete: bool):
        self.med_kernel = parameters["med_kernel"]
        self.savgol_kernel = parameters["savgol_kernel"]
        self.savgol_polyorder = parameters["savgol_polyorder"]
        self.num_bins = parameters["num_bins"]

        if use_discrete:
            self.word_size = parameters["word_size"]
            self.word_stride = parameters["word_stride"]
            self.vocab_size = parameters["vocab_size"]


class PreprocTools:
    def __init__(self):
        self.discretizer = None
        self.vocabulary = None
        self.word_extractor = None
        self.feature_vector_scaler = None
        # self.scaler = None  # TODO probably need a platt scalar for postproc


class DetectorParameters:
    def __init__(self, run_params: RunParams, preproc_params: PreprocParams, model_params: dict, modes):
        self.run_params = run_params
        self.preproc_params = preproc_params
        self.model_params = model_params
        self.modes = modes


class ModeData:
    def __init__(self, detector_params: DetectorParameters, train: bool):
        # self.states = detector_params.run_params.states
        self.detector_params = detector_params
        self.train = train
        self.files = {}
        self.signals = {}
        self.windows = {}
        self.futures = {}

    def add_files(self, file_directory, attack, states=None):
        if states is None:
            states = self.detector_params.run_params.states

        for state in states:
            key = "s" + str(state) + attack

            if key not in self.files:
                self.files[key] = []

            for filename in os.listdir(file_directory):
                if key in filename:
                    self.files[key].append(file_directory + filename)

    def train_test_split(self, split_rules: dict):
        for key in self.files:
            self.files[key].sort()


        for atk in split_rules:
            test_index = split_rules[atk]
            atk_keys = [i for i in self.files.keys() if atk in i]

            for key in atk_keys:

                if self.train:
                    self.files[key] = self.files[key][:test_index]

                else:
                    self.files[key] = self.files[key][test_index:]

    def process_signals_windows_futures(self, use_filter=True):
        """
        Reads csv's specified by ModeData files and returns respective arrays, windows and futures.
        Note that signals are unsuitable for training as individual signals are concatenated with no
        regard for continuity.
        """

        for mode in self.files.keys():
            mode_files = self.files[mode]
            mode_ar = np.empty((0))
            mode_wdws = np.empty((0, self.detector_params.run_params.window_size))
            mode_futures = np.empty((0, self.detector_params.run_params.future_size))

            for file in mode_files:
                ar = pd.read_csv(file)
                ar = np.array(ar).reshape(-1)

                if use_filter:
                    ar = filter_data(
                        ar,
                        self.detector_params.preproc_params.med_kernel,
                        self.detector_params.preproc_params.savgol_kernel,
                        self.detector_params.preproc_params.savgol_polyorder,
                        subsample=1
                    )

                wdws, futures = form_windows(
                    ar,
                    self.detector_params.run_params.window_size,
                    self.detector_params.run_params.window_stride,
                    self.detector_params.run_params.future_size,
                    self.detector_params.run_params.future_stride
                )

                mode_ar = np.concatenate((mode_ar, ar))
                mode_wdws = np.concatenate((mode_wdws, wdws))
                mode_futures = np.concatenate((mode_futures, futures))

            self.signals[mode] = mode_ar
            self.windows[mode] = mode_wdws
            self.futures[mode] = mode_futures



"""
def append_mode_dict(states, test_dict, file_directory, attack, train=True, test_index=0):
    mode_keylist = ["s" + str(state) + "_" + attack for state in states]

    for key in mode_keylist:
        if key not in test_dict:
            test_dict[key] = []

    for filename in os.listdir(file_directory):
        for key in mode_keylist:
            if key in filename:
                test_dict[key].append(file_directory + filename)

    for key in mode_keylist:
        test_dict[key].sort()

        if train:
            test_dict[key] = test_dict[key][:test_index]
        else:
            test_dict[key] = test_dict[key][test_index:]

    return test_dict
"""


def get_mode_data(detector_params: DetectorParameters, split_rules, benign_folder, malicious_folder, train=False):
    mode_data = ModeData(detector_params, train=train)

    file_directory = benign_folder
    mode_data.add_files(file_directory, "_b_")

    file_directory = malicious_folder

    for attack in list(split_rules.keys())[1:]:
        mode_data.add_files(file_directory, attack)

    mode_data.train_test_split(split_rules)
    mode_data.process_signals_windows_futures()

    return mode_data


def filter_data(trace_array, med_kernel=9, savgol_kernel=9, savgol_polyorder=5, subsample=1):
    """Returns Numpy array filtered by median and then Savitzky-Golay filters.
    TODO check parameters here, will likely want to change.
    """
    if savgol_kernel <= savgol_polyorder:
        savgol_polyorder = savgol_kernel - 1

    trace_array = trace_array.reshape(-1)
    trace_array = trace_array[::subsample]
    trace_array = signal.medfilt(trace_array, kernel_size=med_kernel)
    trace_array = signal.savgol_filter(trace_array, savgol_kernel, savgol_polyorder)

    return trace_array


def form_windows(array, window_size, window_stride, future_size=0, future_stride=1):
    """Returns Numpy array of sliding windows of a Numpy array (timeseries).
    https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    """

    last_window = len(array) - window_size - future_size * future_stride + 1

    future_idxs = (
            np.expand_dims(np.arange(0, future_size * future_stride, future_stride), 0)
            + np.expand_dims(np.arange(window_size, window_size + last_window, window_stride), 1)
    )
    futures = array[future_idxs]

    window_idxs = (
            np.expand_dims(np.arange(window_size), 0)
            + np.expand_dims(np.arange(0, last_window, window_stride), 1)
    )
    windows = array[window_idxs]

    return windows, futures


def form_discretizer(array, num_bins=10):
    """
    Returns discretizer fitted to traces of training_dict.
    Note that MultipleCoefficientBinning does not allow for weighting of samples.
    """
    # mcb = approximation.MultipleCoefficientBinning(n_bins=N_BINS, strategy='quantile')
    mcb = approximation.MultipleCoefficientBinning(n_bins=num_bins, strategy="uniform")

    return mcb.fit(array.reshape(-1, 1))


def form_word_extractor(word_size, word_step):
    """Return an instantiated word_extractor."""
    word_extractor = bag_of_words.WordExtractor(word_size, word_step, numerosity_reduction=False)

    return word_extractor


def form_vocab(array, discretizer, word_extractor, vocab_size):
    """Return a vocabulary in list format consisting of the most popular words in the provided count_dict."""
    count_dict = {}
    vocab = []
    words_added = 0

    discrete_array = discretizer.transform(array.reshape(-1, 1))
    words_extracted = word_extractor.transform(discrete_array.reshape(1, -1))
    words_extracted = words_extracted[0].split()

    words, counts = np.unique(words_extracted, return_counts=True)
    for word, count in zip(words, counts):
        if word in count_dict:
            count_dict[word] += count
        else:
            count_dict[word] = count

    while words_added < vocab_size:
        top_word = max(count_dict, key=count_dict.get)
        count_dict[top_word] = 0
        # word_chars = [char for char in top_word]
        # if len(np.unique(word_chars)) > 1:
        vocab.append(top_word)
        words_added += 1

    return vocab


def make_stat_features(wdw_seq):
    """Returns Numpy array statistical features for each provided window.
    TODO change these features to mimic other papers.
    """
    num_features = 8
    fv = np.zeros((len(wdw_seq), num_features))

    wdw_mean = np.mean(wdw_seq, axis=1).reshape(-1)
    wdw_sq = np.square(wdw_seq)
    wdw_rms = np.sqrt(np.sum(wdw_sq, axis=1) / len(wdw_seq))
    wdw_abs_diff = np.abs(np.diff(wdw_seq, axis=1))
    wdw_abs_diff_sum = np.sum(wdw_abs_diff, axis=1)

    fv[:, 0] = wdw_mean
    fv[:, 1] = np.std(wdw_seq, axis=1).reshape(-1)
    fv[:, 2] = stats.skew(wdw_seq, axis=1)
    fv[:, 3] = stats.kurtosis(wdw_seq, axis=1)
    fv[:, 4] = wdw_rms

    fv[:, 4] = np.max(wdw_seq, axis=1).reshape(-1)
    fv[:, 5] = np.min(wdw_seq, axis=1).reshape(-1)
    fv[:, 6] = stats.iqr(wdw_seq, axis=1)
    fv[:, 7] = wdw_abs_diff_sum

    wdw_seq = pd.DataFrame(wdw_seq)
    wdw_seq = wdw_seq.T
    wdw_seq = wdw_seq.mask(wdw_seq.shift(1) == wdw_seq)
    wdw_seq = wdw_seq.T
    wdw_seq = np.array(wdw_seq)
    window_length = int(np.max(np.count_nonzero(~np.isnan(wdw_seq), axis=1)) // 2)

    # estimated entropy of continuous distribution
    # fv[:, 8] = stats.differential_entropy(wdw_seq, axis=1, nan_policy="omit", window_length=100)

    return fv


def make_bow_features(discretizer, vocab, word_extractor, wdw_seq):
    d_wdws = discretizer.transform(wdw_seq.reshape(-1, 1))
    d_wdws = d_wdws.reshape(-1, wdw_seq.shape[1])

    return mp_extract_words(d_wdws, vocab, word_extractor)
    # return extract_words(d_wdws, vocab, word_extractor)


def extract_words(discrete_windows, vocab, word_extractor):
    """Return an array counting vocab occurences for each window of discrete series."""
    count_array = np.zeros((len(discrete_windows), len(vocab)))

    # for wdw_idx in tqdm(range(len(discrete_windows))):
    for wdw_idx in range(len(discrete_windows)):
        wdw = discrete_windows[wdw_idx]
        words_extracted = word_extractor.transform(wdw.reshape(1, -1))
        words_extracted = words_extracted[0].split()
        words, counts = np.unique(words_extracted, return_counts=True)

        for word, count in zip(words, counts):
            try:
                vocab_idx = vocab.index(word)
                count_array[wdw_idx, vocab_idx] = count
            except ValueError:
                pass

    return count_array


def mp_extract_words(discrete_windows, vocab, word_extractor, cores=os.cpu_count()):
    chunksize = len(discrete_windows) // cores
    remainder = len(discrete_windows) % cores

    with Pool(processes=cores) as pool:
        param_list = []
        for i in range(cores):
            tmp_windows = discrete_windows[i * chunksize:(i + 1) * chunksize]
            tmp_params = (tmp_windows, vocab, word_extractor)
            param_list.append(tmp_params)

        results = pool.starmap(extract_words, param_list, chunksize=1)

    result_list = [result for result in results]

    if remainder:
        remainder_windows = discrete_windows[-remainder:]
        remainder_result = extract_words(remainder_windows, vocab, word_extractor)
        result_list.append(remainder_result)

    return np.concatenate(result_list)


def form_preproc_scaler(wdw_seq, discretizer, vocab, word_extractor, use_discrete=True, weight_vector=None):
    """Return a StandardScaler fitted to the X that would be obtained after preprocessing."""
    fv_scaler = StandardScaler()
    fv = make_stat_features(wdw_seq)

    if use_discrete:
        dfv = make_bow_features(discretizer, vocab, word_extractor, wdw_seq)
        fv = np.concatenate((fv, dfv), axis=1)

    return fv_scaler.fit(fv, sample_weight=weight_vector)

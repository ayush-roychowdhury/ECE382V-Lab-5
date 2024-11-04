import sys
sys.path.append("../")

from src.utility import plot_functions, preproc, get_mode_data

import joblib
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.utility.preproc import ModeData, DetectorParameters
from utility import postproc

def tmp_form_mode_data(train: bool):
    mode_data = ModeData(detector_params, train=train)

    split_rules = {
        "_b_": 8, "_m_": 0, "_cc_": 0, "_s_": 0,
    }

    file_dict = {
        data_file_directory: [
            "_b_",
            "_m_", "_cc_", "_s_",
        ]
    }

    for file_directory in file_dict:
        for atk in file_dict[file_directory]:
            mode_data.add_files(file_directory, atk)

    mode_data.train_test_split(split_rules)
    mode_data.process_signals_windows_futures()

    return mode_data


if __name__ == "__main__":
    plt.rcParams['font.size'] = 20

    PLOT_BENIGN_VS_MALWARE = False
    PLOT_ALL_STATES = False

    PRINT_ROC_AUC_SCORES = False
    PLOT_TRACES = False
    PLOT_WORST_ROC = False
    PRINT_MEAN_PROBAS = False


    model = "ocsvm"
    model_dir = model + "/"
    pipeline_name = model
    joblib_dir = "../joblib/" + model_dir
    json_dir = "../json/" + model_dir
    data_file_directory = "../PMD-Dataset/data/"

    pipeline_filename = joblib_dir + pipeline_name + "_detector_pipeline.joblib"
    score_filename = joblib_dir + pipeline_name + "_scores.joblib"

    (proba_dict, class_dict) = joblib.load(score_filename)
    det_pipe = joblib.load(pipeline_filename)
    detector_params = DetectorParameters(det_pipe.run_params, det_pipe.preproc_params, det_pipe.model_params,
                                         det_pipe.modes)

    attack_list = ["_b_", "_m_", "_s_", "_cc_"]

    b_mode_list = detector_params.run_params.states
    m_mode_list = b_mode_list


    # Make note of whether you are plotting/using test or train data
    mode_data = tmp_form_mode_data(train=False)

    if PLOT_BENIGN_VS_MALWARE:
        keylist_0 = ["s2_b_", "s0_m_"]
        keylist_1 = ["s5_b_", "s5_m4_"]
        keylist_1 = ["s5_b_", "s5_m_"]


        fig, ax1 = plt.subplots(2, 1, figsize=(15, 10))

        start = 5000
        end = start + 2000

        trace_start = 0
        for key in keylist_0:
            s1 = mode_data.signals[key][start:end]
            ax1[0].plot(range(trace_start, trace_start + len(s1)), s1, )
            trace_start += len(s1)

        ax1[0].legend(keylist_0)

        trace_start = 0
        for key in keylist_1:
            s1 = mode_data.signals[key][start:end]
            ax1[1].plot(range(trace_start, trace_start+len(s1)), s1,)
            trace_start += len(s1)

        ax1[1].legend(keylist_1)

        ax1[0].set_ylabel("Oscilloscope Reading (mV)")
        ax1[1].set_ylabel("Oscilloscope Reading (mV)")

        plt.show(block=True)

    if PLOT_ALL_STATES:
        def plot_all_benign():
            keylist_0 = [
                "s0_b_",
                "s1_b_",
                "s2_b_",
                "s3_b_",
                "s4_b_",
                "s5_b_",
                "s6_b_",
                "s7_b_",
            ]

            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

            start = 5000
            end = start + 2000

            trace_start = 0
            for key in keylist_0:
                s1 = mode_data.signals[key][start:end]
                ax1.plot(range(trace_start, trace_start + len(s1)), s1, )
                trace_start += len(s1)

            ax1.legend(keylist_0)

            # ax1[1].set_ylim([-0.1, 1.1])
            ax1.set_ylabel("Oscilloscope Reading (mV)")

            # ax1[1].set_xlabel("Sample (Sampling rate 2 kH)")

            plt.show(block=True)

        plot_all_benign()

    # after you train and inference with your model

    if PRINT_ROC_AUC_SCORES:
        for atk_name in attack_list[1:]:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk_name)
            avg_auc = np.format_float_positional(np.mean(roc_array), 3)
            min_auc = np.format_float_positional(np.min(roc_array), 3)
            print(f"{atk_name: <4}:\tavg: {avg_auc}\tmin: {min_auc}")

        print(f"")

    if PLOT_TRACES:

        keylist = ["s4_b_", "s4_cc_", "s5_b_"]
        postproc.plot_traces_and_probabilities(keylist, mode_data, proba_dict)

        # Full plotting
        file_dict = {
            data_file_directory: ["_b_", "_m_", "_cc_", "_s_"],
        }

        # mode_data = ModeData(detector_params, train=True)
        #
        # for file_directory in file_dict:
        #     for atk in file_dict[file_directory]:
        #         mode_data.add_files(file_directory, atk)
        #
        # mode_data.process_signals_windows_futures()
        #
        # keylist = ["s6_b_", "s7_s_"]
        # postproc.plot_traces_and_probabilities(keylist, mode_data, proba_dict)

    if PLOT_WORST_ROC:
        atk_name = "_cc_"
        roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk_name)
        roc_curves = postproc.get_roc_curves(b_mode_list, m_mode_list, proba_dict, atk_name)

        worst_idx = np.argmin(roc_array)
        worst_curve = roc_curves[worst_idx]

        fpr = worst_curve[0]
        tpr = worst_curve[1]
        postproc.print_roc_curve(fpr, tpr, np.min(roc_array))

    if PRINT_MEAN_PROBAS:
        for attack in attack_list:
            keylist = ["s" + str(i) + attack for i in detector_params.run_params.states]
            postproc.print_mean_probas(keylist, proba_dict)
            print(f"")

        print(f"")





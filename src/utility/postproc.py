import matplotlib.pyplot as plt
import numpy as np
import joblib
# import preproc
# from utility import preproc

from sklearn.metrics import mean_squared_error



from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight

from src.utility.preproc import DetectorParameters, ModeData
from src import detector_pipeline


def get_pipeline_scores(detector_pipe: detector_pipeline.DetectorPipeline, mode_data: ModeData, verbose=0):
    proba_dict = {}
    class_dict = {}

    for mode in mode_data.windows:
        window_seq = mode_data.windows[mode]
        proba, class_pred = detector_pipe.score_samples(window_seq)
        proba_dict[mode] = proba
        class_dict[mode] = class_pred

        if verbose > 1:
            print(mode)
            print(f"Probability: {np.mean(proba):6.3f}")
            print(f"Std. Dev.:   {np.std(proba):6.3f}")
            print(f"Class Pred:  {np.mean(class_pred):6.3f}\n")
        elif verbose > 0:
            print(f"{mode}\n")

    return proba_dict, class_dict


def get_pipeline_scores_regression(detector_pipe: detector_pipeline.DetectorPipeline, mode_data: ModeData, verbose=0):
    proba_dict = {}
    pred_dict = {}

    for mode in mode_data.windows:
        window_seq = mode_data.windows[mode]
        y_hat = detector_pipe.score_samples(window_seq)

        future_seq = mode_data.futures[mode]
        y = detector_pipe.preprocessor.transform(future_seq.reshape(-1, 1))
        y = y.reshape(-1, detector_pipe.run_params.future_size)

        loss = mean_squared_error(y.T, y_hat.T, multioutput='raw_values')

        proba_dict[mode] = loss
        pred_dict[mode] = y_hat

        if verbose > 1:
            print(f"\n{mode}")
            print(f"Loss: {np.mean(loss):6.3f}")
            print(f"Mean: {np.mean(y_hat[:, 0]):6.3f}")
        elif verbose > 0:
            print(f"\n{mode}")

    return proba_dict, pred_dict


"""
def get_pipeline_scores(param_filename, pipeline_filename, test_dict, verbose=0):
    parameters = joblib.load(param_filename)
    run_params = parameters[0]
    param_fifo = parameters[1]
    preproc_params = param_fifo[0][0]

    detector_pipe = joblib.load(pipeline_filename)

    _, wdw_dict, _ = preproc.get_signals_windows_futures(run_params, preproc_params, test_dict)

    proba_dict = {}
    class_dict = {}

    for mode in wdw_dict:
        window_seq = wdw_dict[mode]
        proba, class_pred = detector_pipe.score_samples(window_seq)
        proba_dict[mode] = proba
        class_dict[mode] = class_pred

        if verbose > 1:
            print(mode)
            print(f"Probability: {np.mean(proba):6.3f}")
            print(f"Class Pred:  {np.mean(class_pred):6.3f}\n")
        elif verbose > 0:
            print(f"{mode}\n\n")

    return proba_dict, class_dict
"""


def print_mean_probas(keylist, proba_dict):
    for key in keylist:
        probas = proba_dict[key]
        mean = np.mean(probas)
        mean = np.format_float_positional(mean, precision=3)
        print(f"{key}:\t{mean}")


def collate_labels_probas(b_proba, m_proba, benign_positive=True):
    probas = np.concatenate((b_proba, m_proba))
    labels = np.zeros((probas.shape[0]))

    if benign_positive:
        labels[:len(b_proba)] = 1

    else:
        labels[len(b_proba):] = 1

    return labels, probas


def get_roc_curves(b_mode_list, m_mode_list, proba_dict, attack, benign_positive=True):
    roc_curve_list = []

    for i in b_mode_list:
        b_trace = "s" + str(i) + "_b_"
        b_proba = proba_dict[b_trace]

        for j in m_mode_list:
            m_trace = "s" + str(j) + attack
            m_proba = proba_dict[m_trace]

            labels, probas = collate_labels_probas(b_proba, m_proba, benign_positive)
            sample_weights = compute_sample_weight(class_weight='balanced', y=labels)
            fpr, tpr, _ = metrics.roc_curve(labels, probas, sample_weight=sample_weights)
            roc_curve_list.append((fpr, tpr))

    return roc_curve_list


def get_roc_array(b_states, m_states, proba_dict, attack, benign_positive=True):
    roc_array = np.zeros((len(b_states), len(m_states)))

    for i, b_state in enumerate(b_states):
        b_trace = "s" + str(b_state) + "_b_"
        b_proba = proba_dict[b_trace]

        for j, m_state in enumerate(m_states):
            m_trace = "s" + str(m_state) + attack
            m_proba = proba_dict[m_trace]

            labels, probas = collate_labels_probas(b_proba, m_proba, benign_positive)
            sample_weights = compute_sample_weight(class_weight='balanced', y=labels)
            roc_auc = metrics.roc_auc_score(labels, probas, sample_weight=sample_weights)
            roc_array[i, j] = roc_auc

    return roc_array


def plot_traces_and_probabilities(keylist, mode_data: ModeData, proba_dict, figsize=(15, 10)):
    fig, ax1 = plt.subplots(2, 1, figsize=figsize)

    trace_start_a = 0
    trace_start_b = 0
    for key in keylist:
        s1 = mode_data.windows[key][:, 0]
        s2 = proba_dict[key]
        ax1[0].plot(range(trace_start_a, trace_start_a+len(s1)), s1,)
        ax1[1].plot(range(trace_start_b, trace_start_b+len(s2)), 1 - s2)
        trace_start_a += len(s1)
        trace_start_b += len(s2)

    # ax1[1].set_ylim([-0.1, 1.1])
    ax1[0].set_ylabel("Oscilloscope Reading (mV)")
    ax1[1].set_ylabel("Malware Confidence")
    ax1[1].set_xlabel("Sample (Sampling rate 2 kH)")

    plt.show(block=True)
    # fig_filename = output_dir + "/benign_uncolored.png"
    # plt.savefig(fig_filename)


def print_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 8))
    plt.title("Unscaled Data")
    plt.xlabel("Sample")
    plt.ylabel("Voltage (mV)")
    plt.plot(fpr, tpr, color="r", label="ROC curve (area = %0.2f)" % np.min(auc))

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show(block=True)






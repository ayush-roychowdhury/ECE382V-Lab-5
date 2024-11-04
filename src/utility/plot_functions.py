import matplotlib.pyplot as plt
import random

import numpy as np

from sklearn.metrics import mean_squared_error

from src.utility import postproc


def plot_regressor_one_window(wdw_seq, future_seq, preds, idx=None):
    if idx is None:
        idx = random.randint(0, len(wdw_seq))

    future_range = range(len(wdw_seq[0]), len(wdw_seq[0]) + len(future_seq[0]))
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
    ax1.plot(wdw_seq[idx, :], color="blue")
    ax1.plot(future_range, future_seq[idx, :], color="green")
    ax1.plot(future_range, preds[idx, :], color="red")

    plt.show()


def plot_regressor_loss(future_seq, preds):
    loss = mean_squared_error(future_seq.T, preds.T, multioutput='raw_values')

    fig, ax1 = plt.subplots(2, 1, figsize=(15, 10))
    ax1[0].plot(future_seq[:, 0], color="blue")
    ax1[0].plot(preds[:, 0], color="red")

    ax1[1].plot(loss, color="green")

    plt.show()


def plot_regressor_predictions(future_seq, preds):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
    ax1.plot(future_seq[:, 0], color="blue")
    ax1.plot(preds[:, 0], color="red")
    plt.show()


def print_slowdown_attacks(proba_dict, b_mode_list, m_mode_list):
    slowdown_attacks = [
        ["_m_", "_m4_", "_m16_"],
        ["_cc_", "_cc4_", "_cc16_"],
        ["_s_", "_s4_", "_s16_"],
    ]

    for slowed_attacks in slowdown_attacks:
        for atk_name in slowed_attacks:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk_name)
            roc_array = roc_array.reshape(-1)
            print(f"{atk_name}:\tavg: {np.mean(roc_array):5.3f}\tmin: {np.min(roc_array):5.3f}")
        print(f"")


def plot_slowdown_attacks(proba_dict, b_mode_list, m_mode_list):
    slowdown_attacks = [
        ["_m_", "_m4_", "_m16_"],
        ["_cc_", "_cc4_", "_cc16_"],
        ["_s_", "_s4_", "_s16_"],
    ]

    roc_means = []
    y_error = []

    for i, atk_list in enumerate(slowdown_attacks):
        tmp_means = []
        tmp_mins = []
        tmp_maxs = []

        for atk in atk_list:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk)
            roc_array = roc_array.reshape(-1)
            tmp_means.append(np.mean(roc_array))
            tmp_mins.append(np.mean(roc_array) - np.min(roc_array))
            tmp_maxs.append(np.max(roc_array) - np.mean(roc_array))

        roc_means.append(tmp_means)
        y_error.append([tmp_mins, tmp_maxs])

    attack_labels = ["Meltdown", "L1 Covert-Channel", "Spectre"]

    fig, ax1 = plt.subplots(1, 3, figsize=(15, 10))
    for i in range(3):
        ax1[i].errorbar(np.arange(3), roc_means[i], yerr=y_error[i], xerr=None, fmt='o', markersize=12, lw=3,
                        markerfacecolor="black", color="red")

    ax1[0].set_ylim([0.4, 1.02])
    ax1[1].set_ylim([0.4, 1.02])

    for i in range(3):
        ax1[i].set_ylim([0.4, 1.02])
        ax1[i].set_xticks([0, 1, 2])  # Set x-axis ticks for the second subplot
        ax1[i].set_xticklabels(['x1', 'x4', 'x16'])
        ax1[i].text(0.15, 0.41, attack_labels[i], ha='left', va='center')

    # fig.supxlabel('Slowdown Rates', )
    # fig.supylabel('ROC-AUC')
    fig.text(0.5, 0.03, 'Slowdown Rates', ha='center', va='center')
    fig.text(0.02, 0.5, 'ROC-AUC', ha='center', va='center', rotation='vertical')

    plt.tight_layout(pad=3)
    plt.show(block=True)


def plot_slowdown_attacks_boxplot(proba_dict, b_mode_list, m_mode_list):
    slowdown_attacks = [
        ["_m_", "_m2_", "_m4_", "_m8_", "_m16_", "_m32_"],
        ["_cc_", "_cc2_", "_cc4_", "_cc8_", "_cc16_", "_cc32_"],
        ["_s_", "_s2_", "_s4_", "_s8_", "_s16_", "_s32_"],
    ]

    roc_boxes = []

    for i, atk_list in enumerate(slowdown_attacks):
        roc_atk_box = []

        for atk in atk_list:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk)
            roc_array = roc_array.reshape(-1)
            roc_atk_box.append(roc_array)

        roc_boxes.append(roc_atk_box)

    attack_labels = ["Meltdown", "L1 Covert-Channel", "Spectre"]

    flierprops = dict(marker='o', markerfacecolor='red', markersize=12, linestyle='none', alpha=0.5)

    fig, ax1 = plt.subplots(1, 3, figsize=(16, 10))
    for i in range(3):
        box = ax1[i].boxplot(roc_boxes[i], patch_artist=False, flierprops=flierprops)

        for median in box['medians']:
            median.set(color='blue', linewidth=3)

    ax1[0].set_ylim([0.4, 1.02])
    ax1[1].set_ylim([0.4, 1.02])

    for i in range(3):
        ax1[i].set_ylim([0.4, 1.02])
        ax1[i].set_xticks([1, 2, 3, 4, 5, 6])  # Set x-axis ticks for the second subplot
        ax1[i].set_xticklabels(['1', '2', '4', '8', '16', '32'])
        ax1[i].text(0.6, 0.41, attack_labels[i], ha='left', va='center')

    fig.text(0.5, 0.03, 'Slowdown Rates', ha='center', va='center')
    fig.text(0.02, 0.5, 'ROC-AUC', ha='center', va='center', rotation='vertical')

    plt.tight_layout(pad=3)
    plt.show(block=True)


def get_mitre_attack_names():
    # initial_access_atks = ["_ssh_", "_scp_", "_ufw_", "_fd_"]
    # execution_atks = ["_user_", "_enum_cfg_", "_enum_net_", "_enum_prot_", "_enum_sys_", "_enum_user_hist_", "_hist_"]
    # exfil_atks = ["_mp_", "_nmap_", "_perm_", "_ps_", "_psswd_"]
    # mitre_attacks = [initial_access_atks, execution_atks, exfil_atks]

    attack_overhead = [
        "_ssh_", "_scp_", "_ufw_", "_fd_",
        "_nmap_", "_perm_", "_ps_", "_psswd_",
    ]
    execution_atks = [
        "_user_", "_enum_cfg_", "_enum_net_", "_enum_prot_", "_enum_sys_", "_enum_user_hist_", "_hist_", "_mp_",
    ]

    mitre_attacks = [execution_atks, attack_overhead]

    return mitre_attacks


def print_mitre_attack_metrics(proba_dict, b_mode_list, m_mode_list):
    mitre_attacks = get_mitre_attack_names()

    for attack_list in mitre_attacks:
        for atk_name in attack_list:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk_name)
            roc_array = roc_array.reshape(-1)
            print(f"{atk_name}:\tavg: {np.mean(roc_array):5.3f}\tmin: {np.min(roc_array):5.3f}")
        print(f"")


def plot_mitre_attacks(proba_dict, b_mode_list, m_mode_list):
    mitre_attacks = get_mitre_attack_names()

    for attack_list in mitre_attacks:

        roc_means = []
        roc_mins = []
        roc_maxs = []

        for atk in attack_list:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk)
            roc_array = roc_array.reshape(-1)
            roc_means.append(np.mean(roc_array))
            roc_mins.append(np.mean(roc_array) - np.min(roc_array))
            roc_maxs.append(np.max(roc_array) - np.mean(roc_array))

        y_error = [roc_mins, roc_maxs]

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

        ax1.errorbar(np.arange(len(attack_list)), roc_means, yerr=y_error, xerr=None, fmt='o', markersize=12, lw=3,
                     markerfacecolor="black", color="red")

        ax1.set_xticks([i for i in range(len(attack_list))])  # Set x-axis ticks for the second subplot
        ax1.set_xticklabels([atk for atk in attack_list], rotation=90)

        fig.text(0.02, 0.5, 'ROC-AUC', ha='center', va='center', rotation='vertical')

        plt.tight_layout(pad=3)
        plt.show(block=True)


def plot_mitre_attacks_boxplot(proba_dict, b_mode_list, m_mode_list):
    mitre_attacks = get_mitre_attack_names()

    for attack_list in mitre_attacks:

        roc_boxes = []

        for atk in attack_list:
            roc_array = postproc.get_roc_array(b_mode_list, m_mode_list, proba_dict, atk)
            roc_array = roc_array.reshape(-1)
            roc_boxes.append(roc_array)

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

        flierprops = dict(marker='o', markerfacecolor='red', markersize=12, linestyle='none', alpha=0.5)
        box = ax1.boxplot(roc_boxes, patch_artist=False, flierprops=flierprops)

        for median in box['medians']:
            median.set(color='blue', linewidth=3)

        ax1.set_xticks([i + 1 for i in range(len(attack_list))])  # Set x-axis ticks for the second subplot
        ax1.set_xticklabels([atk for atk in attack_list], rotation=90)

        fig.text(0.02, 0.5, 'ROC-AUC', ha='center', va='center', rotation='vertical')

        plt.tight_layout(pad=3)
        plt.show(block=True)


def plot_target_complexity(model_score_dict):
    roc_means = []
    y_error = []

    for model in model_score_dict:
        tmp_means = []
        tmp_mins = []
        tmp_maxs = []

        for run_type in model_score_dict[model]:
            roc_array = np.array(model_score_dict[model][run_type]).reshape(-1)
            tmp_means.append(np.mean(roc_array))
            tmp_mins.append(np.mean(roc_array) - np.min(roc_array))
            tmp_maxs.append(np.max(roc_array) - np.mean(roc_array))

        roc_means.append(tmp_means)
        y_error.append([tmp_mins, tmp_maxs])


    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    num_models = len(model_score_dict)
    base_ticks = np.array([0, 1.5, 3, 5, 6.5, 8]) * num_models
    base_tick_labels = ['1-Benign', '2-Benign', '3-Benign', '1-Benign', '2-Benign', '3-Benign']
    hidden_ticks = [base_ticks[1], base_ticks[4]]
    hidden_tick_labels = ["Single-Core", "Multi-Core"]

    num_models = len(model_score_dict)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i / (num_models - 1)) for i in range(num_models)]

    for i, model in enumerate(model_score_dict):
        ax1.errorbar(base_ticks + i, roc_means[i], yerr=y_error[i], xerr=None, fmt='o', markersize=12, lw=3,
                            markerfacecolor=colors[i], color=colors[i], label=model)

    ax1.set_ylim([-0.1, 1.1])
    ax1.set_xticks(base_ticks)
    ax1.set_xticklabels(base_tick_labels)
    ax1.legend(loc='lower left')

    # Set x-axis ticks for the second subplot
    ax2 = fig.add_subplot(111, label="secondary")
    ax2.set_facecolor("none")
    # ax2.set_aspect("equal")
    ax2.set_xticks(hidden_ticks)
    ax2.set_xticklabels(hidden_tick_labels)
    ax2.tick_params(axis='x', pad=30)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=True,
                    top=False, labeltop=False)
    ax2.set_facecolor("none")
    for _, spine in ax2.spines.items():
        spine.set_visible(False)

    plt.tight_layout(pad=3)
    plt.show(block=True)


def plot_target_complexity_box(model_score_dict, ylim=[-0.1, 1.1], pad=3):
    roc_boxes = []

    for model in model_score_dict:
        model_roc_boxes = []

        for run_type in model_score_dict[model]:
            roc_array = np.array(model_score_dict[model][run_type]).reshape(-1)
            roc_array = roc_array.reshape(-1)
            model_roc_boxes.append(roc_array)

        roc_boxes.append(model_roc_boxes)


    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7.5))

    num_models = len(model_score_dict)
    base_ticks = np.array([0, 1.5, 3, 5, 6.5, 8]) * num_models
    base_ticks += 1
    label_ticks = base_ticks + num_models // 2 - 0.5
    label_tick_labels = ['1-Benign', '2-Benign', '3-Benign', '1-Benign', '2-Benign', '3-Benign']
    hidden_ticks = [base_ticks[1] + num_models // 2, base_ticks[4] + num_models // 2]
    hidden_tick_labels = ["Single-Core", "Multi-Core"]

    num_models = len(model_score_dict)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i / (num_models - 1)) for i in range(num_models)]

    for i, model in enumerate(model_score_dict):
        flierprops = dict(marker='o', markerfacecolor='red', markersize=12, linestyle='none', alpha=0.5)
        box = ax1.boxplot(roc_boxes[i], positions=base_ticks + i, patch_artist=True, flierprops=flierprops, label=model)

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

        for median in box['medians']:
            median.set(color="red", linewidth=3)

    ax1.set_xlim([0, base_ticks[-1] + num_models])
    ax1.set_ylim(ylim)
    # ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_yticks([0.8, 0.85, 0.90, 0.95, 1])
    ax1.set_xticks(label_ticks)
    ax1.set_xticklabels(label_tick_labels)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  ncol=len(model_score_dict), prop={'size': 20})
    ax1.set_ylabel("ROC-AUC Scores")

    # Set x-axis ticks for the second subplot
    ax2 = fig.add_subplot(111, label="secondary")
    ax2.set_facecolor("none")
    # ax2.set_aspect("equal")
    ax2.set_xticks(hidden_ticks)
    ax2.set_xticklabels(hidden_tick_labels)
    ax2.tick_params(axis='x', pad=30)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=True,
                    top=False, labeltop=False)
    # , labelsize=28)
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    ax2.set_facecolor("none")
    for _, spine in ax2.spines.items():
        spine.set_visible(False)

    plt.tight_layout(pad=pad)
    plt.show(block=True)


def plot_prior_works(model_score_dict):
    roc_means = []
    y_error = []

    for model in model_score_dict:
        tmp_means = []
        tmp_mins = []
        tmp_maxs = []

        for run_type in model_score_dict[model]:
            roc_array = np.array(model_score_dict[model][run_type]).reshape(-1)
            tmp_means.append(np.mean(roc_array))
            tmp_mins.append(np.mean(roc_array) - np.min(roc_array))
            tmp_maxs.append(np.max(roc_array) - np.mean(roc_array))

        roc_means.append(tmp_means)
        y_error.append([tmp_mins, tmp_maxs])

    roc_means = []
    y_error = []

    for model in model_score_dict:
        tmp_means = []
        tmp_mins = []
        tmp_maxs = []

        for run_type in model_score_dict[model]:
            roc_array = np.array(model_score_dict[model][run_type]).reshape(-1)
            tmp_means.append(np.mean(roc_array))
            tmp_mins.append(np.mean(roc_array) - np.min(roc_array))
            tmp_maxs.append(np.max(roc_array) - np.mean(roc_array))

        roc_means.append(tmp_means)
        y_error.append([tmp_mins, tmp_maxs])

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    base_ticks = np.array([0]) * 3
    base_tick_labels = ['3-Benign Singlecore']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, model in enumerate(model_score_dict):
        ax1.errorbar(base_ticks + i, roc_means[i], yerr=y_error[i], xerr=None, fmt='o', markersize=12, lw=3,
                     markerfacecolor=colors[i], color=colors[i], label=model)

    ax1.set_ylim([-0.1, 1.1])
    # ax1.set_xticks([len(model_score_dict) / 2])
    # ax1.set_xticklabels(base_tick_labels)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax1.set_ylabel('ROC-AUC')
    ax1.set_xlabel('3-Benign Singlecore')
    ax1.legend(loc='lower right')

    plt.tight_layout(pad=3)
    plt.show(block=True)


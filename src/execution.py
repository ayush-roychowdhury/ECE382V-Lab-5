
import sys
sys.path.append("../")

from src.utility.preproc import DetectorParameters, ModeData

from src import detector_pipeline
import joblib
import json
import time
from sklearn.model_selection import ParameterSampler

from utility import preproc
from utility import postproc



def top_training():
    def form_detector_params(pipeline_filename: str):
        PARAM_SAMPLES = 2
        parameter_idx = 0

        param_grid_filename = json_dir + "../" + "preproc_param_grid.json"
        model_grid_filename = json_dir + pipeline_name + "_model_grid.json"

        with open(param_grid_filename) as json_file:
            preproc_param_grid = json.load(json_file)

        with open(model_grid_filename) as json_file:
            model_param_grid = json.load(json_file)

        for key in model_param_grid:
            for i, item in enumerate(model_param_grid[key]):
                if item == "None":
                    model_param_grid[key].insert(i, None)
                    model_param_grid[key].remove("None")

        run_params = preproc.RunParams()
        run_params.states = [i for i in range(8)]
        run_params.window_size = 999  # TODO need to manually set sampling rate back to 1 KHz
        run_params.window_stride = SCORE_STRIDE
        run_params.future_size = 3
        run_params.future_stride = 2
        run_params.use_discrete = True
        run_params.train_new = True  # TODO do something about this

        preproc_param_list = list(ParameterSampler(preproc_param_grid, n_iter=PARAM_SAMPLES, random_state=0))
        model_param_list = list(ParameterSampler(model_param_grid, n_iter=PARAM_SAMPLES, random_state=0))

        preproc_params_dict = preproc_param_list[parameter_idx]
        preproc_params = preproc.PreprocParams(preproc_params_dict, run_params.use_discrete)
        model_params = model_param_list[parameter_idx % len(model_param_list)]

        classes = ["s" + str(state) + "_b_" for state in run_params.states]
        detector_params = DetectorParameters(run_params, preproc_params, model_params, classes)

        return detector_params

    pipeline_filename = joblib_dir + pipeline_name + "_detector_pipeline.joblib"

    detector_params = form_detector_params(pipeline_filename)
    detector_params.run_params.window_stride = TRAIN_STRIDE

    mode_data = ModeData(detector_params, train=True)
    file_directory = data_file_directory
    mode_data.add_files(file_directory, "_b_")
    mode_data.train_test_split({"_b_": 8})
    mode_data.process_signals_windows_futures()

    det_pipe = detector_pipeline.get_model(model, detector_params)
    det_pipe.train_preprocessor(mode_data)
    det_pipe.train_detector(mode_data)

    print(" *** Done Training")

    joblib.dump(det_pipe, pipeline_filename)


def top_score_samples():
    pipeline_filename = joblib_dir + pipeline_name + "_detector_pipeline.joblib"
    score_filename = joblib_dir + pipeline_name + "_scores.joblib"

    det_pipe = joblib.load(pipeline_filename)
    detector_params = DetectorParameters(det_pipe.run_params, det_pipe.preproc_params, det_pipe.model_params,
                                         det_pipe.modes)

    detector_params.run_params.window_stride = SCORE_STRIDE

    split_rules = {"_b_": 8, "_m_": 4, "_cc_": 4, "_s_": 4}
    mode_data = preproc.get_mode_data(detector_params, split_rules, data_file_directory,
                                      data_file_directory, train=False)

    proba_dict, class_dict = postproc.get_pipeline_scores(det_pipe, mode_data, 2)

    scores_data = (proba_dict, class_dict)
    joblib.dump(scores_data, score_filename)

    print(" *** Done Scoring")


if __name__ == "__main__":
    start_time = time.time()

    TRAIN_NEW = True
    SCORE_SAMPLES = True

    TRAIN_STRIDE = 100  # 10
    SCORE_STRIDE = 250  # 50

    model = "ocsvm"
    model_dir = model + "/"
    pipeline_name = model
    joblib_dir = "../joblib/" + model_dir
    json_dir = "../json/" + model_dir
    data_file_directory = "../PMD-Dataset/data/"

    if TRAIN_NEW:
        top_training()

    if SCORE_SAMPLES:
        top_score_samples()

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {minutes:5.0f}:{seconds:0.2f}")

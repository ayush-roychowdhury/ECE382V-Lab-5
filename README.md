# ECE382V-Lab-5
Lab 5 (ML for Security)

# use python 3.10

# [1] Clone repo, then clone PMD-dataset as submodule into repo

# [2] run eval.py with the PLOT_BENIGN_VS_MALWARE and PLOT_ALL_STATES toggles set to True. Other toggles set to false.
#  First plot shows benign vs. malware power trace. As we can see, first case is easy to differentiate by eye.
#  Second case is hard. Can we use ML to help us?
#  Second plot shows all 8 operating modes of system. Operating mode defined as unique combination of executing states.
#  For more detail, look at dataset.

# [3] Go to execution.py. Read documentation on detector. Detector is ensemble of one class classification pipelines.
#  TODO include the documentation.
#  Key idea is to fit OCC to each benign mode. Everything else is labelled malware.
#  Run execution.py to train and inference detector.
#  You can implement your own detector here. Lots of optimization opportunity.

# [4] Go back to eval.py
#  Evaluate your detector performance with ROC-AUC metric.
#  Other toggles to help with your analysis.


File Descriptions:

Files: the original CSV files pulled from the UN
Sets: the fully processed data CSV files

preprocess.py: the script used to populate Sets, note that to run this script, the target directory MUST be changed in the file

analysis_demonstration.py: the script used to demonstrate how prob_cloud works. this script can be run from the command line with $ python preprocess.py

prob_cloud: a regressor that attempts to find the local point densities to approximate the probability distribution, and then 
	performs the prediction by sampling potential values and selecting the best predictions based on artificial "mean", "median", and "mode" approximations

geneticSelect: a helper function that uses genetic sampling to pick the best features to perform regression with - designed to be implemented in the future with the iterative functions

analysisIterative.py: these new functions are attempts to apply iterative uses of prob_cloud to perform dimension reduction by trying to predict things like residules
	between marginal distributions. These are still being tested and refined.
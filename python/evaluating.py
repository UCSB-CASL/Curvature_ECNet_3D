"""
Evaluate trained k_ecnets on various surfaces.  To do this, we load the networks for non-saddles and saddles with their corresponding
standard and pca scalers.  The data sets for offline evaluation already include each sample normalized by h (for the level-set field).  For
numerical saddles (i.e., ih2kg < config.NON_SADDLE_MIN_IH2KG) we don't perform negative-curvature normalization, and we don't combine the
numerical estimation with the neural response.  For non-saddle samples (i.e., ih2kg >= config.NON_SADDLE_MIN_IH2KG), these are already
normalized to the negative curvature spectrum; here, we use the neural network only if |ihk| >= config.MIN_HK; upon inference, we combine
the neural response with the numerical estimation smoothly if ihk is near 0.  Whether we are dealing with saddle or non-saddle samples,
these are already in standard-form and the (possibly negated) gradient at the center grid node has all its components positive.

Recall, for each point next to Gamma, there are six standard-formed samples.

Created: August 5, 2022.
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn import metrics
from typing import List
import importlib
import logging
import os
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils

importlib.reload( utils )

############################################################################################################################################
"""
TODO: Change these configuration parameters.
"""
MAX_RL = 6											# Desired maximum level of refinement.

# File paths.
DATA_ROOT = "../data/" + str( MAX_RL ) + "/"		# Data sets root folder.
MODELS_ROOT = "../models/" + str( MAX_RL ) + "/"	# Where to store models and preprocessing objects.
RESULTS_ROOT = "../results/" + str( MAX_RL ) + "/"	# Where to store results.

############################################################################################################################################

# Parameters used in casl_p4est too.
H = 1 / 2 ** MAX_RL				# Mesh size.
P4EST_DIM = 3
NUM_NEIGHBORS_CUBE = 3 ** 3
K_INPUT_SIZE_LEARN = 112		# Includes two additional slots for true curvatures.

MAX_HK = 2./3					# Max mean |hk| (across resolutions) for non-saddle regions.
MIN_HK = 0.004					# Min mean |hk| (across resolutions) for non-saddle regions.
NON_SADDLE_MIN_IH2KG = -7e-6	# Min ih2kg to consider a sample from a non-saddle region.
MAX_SADDLE_HK = 0.43			# Max |ihk| we can reliable infer from for saddle regions.

LO_MIN_HK = MIN_HK
UP_MIN_HK = 0.007
BATCH_SIZE = 64					# Batch size for training and inference with tensorflow.
N_PERMUTS = 6

def predict( model: keras.models.Sequential, pcaScaler: PCA, stdScaler: StandardScaler, inputs: List[np.ndarray], ihk: np.ndarray ) -> np.ndarray:
	"""
	Compute the neural prediction by scaling inputs and averaging outputs from the N_PERMUTS data-packet permutations in standard-form.
	:param model: Neural network.
	:param pcaScaler: PCA scaler.
	:param stdScaler: Standard scaler.
	:param inputs: Six standard-formed inputs.
	:param ihk: Numerical hk interpolated at the interface.
	:return: Average prediction.
	"""
	nnPred = np.zeros( len( ihk ) )
	for ins in inputs:
		ins = stdScaler.transform( ins.astype( np.float64 ) ).astype( np.float32 )
		ins = pcaScaler.transform( ins.astype( np.float64 ) ).astype( np.float32 )
		nnPred = nnPred + model.predict( [ins, ihk], batch_size=BATCH_SIZE ).flatten()

	nnPred = nnPred / len( inputs )  # Leverage curvature reflection/rotation invariance by averaging predictions.
	return nnPred


def evaluate( shapeType: str, experimentId: int, reinitIter: int, useNnet: bool=True ):
	"""
	Evaluate best model's performance on a surface (e.g., ellipsoid, paraboloid, Gaussian).
	:param shapeType: Type of surface.
	:param experimentId: Experiment id.
	:param reinitIter: Number of redistancing iterations.
	:param useNnet: Whether to use neural network.  If so, show results for both numerical and inferred hk values.
	"""
	global DATA_ROOT
	DATA_ROOT = DATA_ROOT + shapeType + "/" + str( experimentId ) + "/"	# E.g., "../data/6/ellipsoid/0/"

	# Read in parameters and samples.
	prefix = "iter" + str( reinitIter )
	params: pd.DataFrame = pd.read_csv( DATA_ROOT + prefix + "_params.csv" )
	data: np.ndarray = pd.read_csv( DATA_ROOT + prefix + "_data.csv" ).to_numpy( np.float32 )

	if len( data ) % N_PERMUTS != 0 or len( data ) == 0:
		raise "Number of samples must be a multiple of {}.".format( N_PERMUTS )

	logging.info( "------------- Evaluating {:^10}, experiment id {} -------------".format( shapeType, experimentId ) )
	logging.info( "* MaxRefLevel: {}".format( MAX_RL ) )
	logging.info( "* Sampled points: {}".format( len( data ) // N_PERMUTS ) )
	logging.info( "* Parameters:" )
	for c in params.columns:
		logging.info( "  {}: {}".format( c, params[c][0] ) )
	logging.info( "------------------------------------------------------------------" )

	data, targets = utils.buildInputsAndTargets( data )		# Split inputs and targets (including separate ihk input).  Get rid of h2kg.
	ihk: np.ndarray = targets[:, -1]						# Numerical hk to fix with neural network.
	hk = targets[:, 0]										# Target dimensionless mean curvature.
	del targets

	splitData = list()
	for i in range( N_PERMUTS ):							# Data sets have N_PERMUTS samples per point--all in standard form.
		splitData.append( data[i::N_PERMUTS] )
	ihk = ihk[0::N_PERMUTS]; hk = hk[0::N_PERMUTS]			# ihk and hk stay the same in any sample for the same point.
	del data

	# Statistical analyses.
	if useNnet:
		# Retrieve the best models and preprocessing scalers.
		model: List[keras.models.Sequential] = list()		# model[0] for non-saddles and model[1] for saddles.
		model.append( keras.models.load_model( MODELS_ROOT + "non-saddle/k_nnet.h5" ) )
		model.append( keras.models.load_model( MODELS_ROOT + "saddle/k_nnet.h5" ) )
		pcaScaler: List[PCA] = list()								# pca_scaler[0] for non-saddles and pca_scaler[1] for saddles.
		pcaScaler.append( pk.load( open( MODELS_ROOT + "non-saddle/k_pca_scaler.pkl", "rb" ) ) )
		pcaScaler.append( pk.load( open( MODELS_ROOT + "saddle/k_pca_scaler.pkl", "rb" ) ) )
		stdScaler: List[StandardScaler] = list()					# std_scaler[0] for non-saddles and std_scaler[1] for saddles.
		stdScaler.append( pk.load( open( MODELS_ROOT + "non-saddle/k_std_scaler.pkl", "rb" ) ) )
		stdScaler.append( pk.load( open( MODELS_ROOT + "saddle/k_std_scaler.pkl", "rb" ) ) )

		# Split data into non-saddles and saddles (use the last column to do this: ih2kg).
		# We can use any entry in splitData since ih2kg is identical for all samples of the same point.
		idx: List[List[int]] = list()
		idx.append( np.where( splitData[0][:,-1] >= NON_SADDLE_MIN_IH2KG )[0].tolist() )	# Non-saddle indices.
		idx.append( np.where( splitData[0][:,-1] < NON_SADDLE_MIN_IH2KG )[0].tolist() )		# Saddle indices.

		if len( idx[0] ) == 0:
			logging.warning( "++ There are no non-saddle samples!" )
		if len( idx[1] ) == 0:
			logging.warning( "++ There are no saddle samples!" )

		# Perform inference: first on non-saddles, and then on saddles.
		predictions: np.ndarray = np.copy( ihk )
		for i in range( 2 ):
			inputs: List = list()
			for j in range( N_PERMUTS ):
				inputs.append( splitData[j][idx[i]] )	# Organize all permutations.  All these are copies because idx is a list (not a view).
			lihk = ihk[idx[i]]			# Local copy of ihk.
			nnIdx = []; nnPred = []

			if i == 0 and len(idx[i]) > 0:	# Non-saddle regions.
				nnIdx = np.where( np.abs( lihk ) >= LO_MIN_HK )[0].tolist()		# Where to use the neural network.
				for j in range( N_PERMUTS ):
					inputs[j] = inputs[j][nnIdx]
					# inputs[j][:,-1] = np.abs( inputs[j][:, -1] )  # Taking |ih2kg|.  Do I have to take abs if I didn't use this when training?
				nnihk = lihk[nnIdx]			# ihk where we'll use nnet.
				sign = np.sign( nnihk )		# ihk signs of rows where we'll use the nnet.

				nnihk = -np.abs( nnihk )	# Normalize to negative ihk.
				for j in range( N_PERMUTS ):
					inputs[j][:,-2] = -np.abs( inputs[j][:,-2] )

				nnPred = predict( model[i], pcaScaler[i], stdScaler[i], inputs, nnihk )
				blendIdx = np.where( np.abs( nnihk ) <= UP_MIN_HK )[0].tolist()			# Neural indices where we can blend with numerical estimation near 0.
				lam = (UP_MIN_HK - np.abs( nnihk[blendIdx] )) / (UP_MIN_HK - LO_MIN_HK) # Convex-combination coefficient based on ihk.
				nnPred[blendIdx] = (1 - lam) * nnPred[blendIdx] + lam * nnihk[blendIdx]
				nnPred = sign * np.abs( nnPred )										# Restore sign for neural predictions.

			elif i == 1 and len(idx[i])>0:	# Saddle regions.
				nnIdx = np.where( np.abs( lihk ) <= MAX_SADDLE_HK )[0].tolist()			# We only have a reliable range for ihk to use saddles' nnet.
				if len( nnIdx ) < len( lihk ):
					logging.warning( "++ There are {} SADDLE samples out of {} where the neural network won't be used!"
									 .format( len( lihk ) - len( nnIdx ), len( lihk ) ) )
				for j in range( N_PERMUTS ):
					inputs[j] = inputs[j][nnIdx]
				nnihk = lihk[nnIdx]  		# ihk where we'll use nnet.

				nnPred = predict( model[i], pcaScaler[i], stdScaler[i], inputs, nnihk )

			lihk[nnIdx] = nnPred
			predictions[idx[i]] = lihk	# Put predictions back into array.

		ylabel = r"Inferred $h\kappa^\star$"
		utils.plotCorrelation( hk, predictions, RESULTS_ROOT + "iter" + str( reinitIter ) + "_hybrid1.png", "goldenrod", ylabel, r"h\kappa^\star",
							   True, False, idx, ["tab:blue", "tab:red"], ["Non-saddle data", "Saddle data"])

		# Collect stats for hybrid computations.
		hybMAE = metrics.mean_absolute_error( hk, predictions )
		hybMaxAE = max( np.abs( hk - predictions ) )
		hybRMSE = np.sqrt( metrics.mean_squared_error( hk, predictions ) )
		logging.info( "  Method                     | Mean Absolute Error   | Max Absolute Error    | Root Mean Sqrd Error" )
		logging.info( "* Neural model on iter{}     | {:>20.10e}  | {:>20.10e}  | {:>20.10e}".format( reinitIter, hybMAE, hybMaxAE, hybRMSE ) )

	# Numerical errors.
	utils.plotCorrelation( hk, ihk, RESULTS_ROOT + "iter" + str( reinitIter ) + "_numerical1.png", "C2", r"Numerical $h\kappa$", r"h\kappa" )

	numMAE = metrics.mean_absolute_error( hk, ihk )
	numMaxAE = max( np.abs( hk - ihk ) )
	numRMSE = np.sqrt( metrics.mean_squared_error( hk, ihk ) )
	if not useNnet:
		logging.info( "  Method                     | Mean Absolute Error   | Max Absolute Error    | Root Mean Sqrd Error" )
	logging.info( "* Numerical method on iter{} | {:>20.10e}  | {:>20.10e}  | {:>20.10e}".format( reinitIter, numMAE, numMaxAE, numRMSE ) )


################################################################ Evaluation ################################################################
"""
TODO: Comment/uncomment pairs of lines below to change the experiment with ellipsoid, Gaussian, paraboloid, or hyperbolic paraboloid.
"""

if __name__ == "__main__":
	# Ellipsoid experiments.
	shape = "ellipsoid"
	eId = 0; reinitIters = 10; withNnet = True  	# Level  6: random=11, a=1.65, b=0.75, c=0.2, hka=0.345182, hkb=0.148637, hkc=0.003352.

	# Gaussian experiments.
	# shape = "gaussian"
	# eId = 0; reinitIters = 10; withNnet = True  	# Level  6: random=7, a=1, su/sv=3, max_hk=0.6.

	# Paraboloid experiments.
	# shape = "paraboloid"
	# eId = 0; reinitIters = 10; withNnet = True 	# Level  6: random=7, a=25.6, b=12.8, c=0.5, hk=0.6.

	# Hyperbolic paraboloid experiments.
	# shape = "hyp_paraboloid"
	# eId = 0; reinitIters = 10; withNnet = True 	# Level  6: random=11, a=33.6, b=11.2, hk=0.35.

	RESULTS_ROOT = RESULTS_ROOT + shape + "/" + str( eId ) + "/"		# E.g., "../results/6/ellipsoid/0/"
	rootLogger = utils.setUpRootLogger( RESULTS_ROOT + "iter_" + str( reinitIters ) + "_" + os.path.basename( __file__ ) + "1.log" )
	evaluate( shape, eId, reinitIters, withNnet )
	rootLogger.handlers = []											# Release root logging handlers.
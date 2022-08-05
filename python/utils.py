"""
Utility functions.
Created: March 28, 2022.
Updated: April 16, 2022.
"""

import logging
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from typing import List, Union


def buildInputsAndTargets( ds: np.ndarray ) -> (np.ndarray, np.ndarray):
	"""
	Build the inputs and target fields for a learning/offline-evaluation data set by splitting ds into a subset with 110 components
	(i.e., phi/h, u, v, w, ihk, ih2kg) as inputs and another with 2 components (i.e., hk, ihk).  In the latter, we'll extract ihk as a sepa-
	rate input to the neural network.
	:param ds: 112-column numpy matrix data set to split.
	:return: A 110-column matrix as general inputs, and a 2-column matrix as targets.
	"""
	tgts: np.ndarray = np.copy( ds[:, -4:] )
	ds = ds[:, :-4]
	ds = np.c_[ds, np.copy( tgts[:, [-3, -1]] )]	# Copy ihk and ih2kg back to full data set (in that order).
	return ds, tgts[:, :-2]  						# Targets contains only hk and ihk (not h2kg and ih2kg) in that order.


def setUpRootLogger( fileName: str ):
	"""
	Set up the root logger to print to a log file and to the standard output.
	:param fileName: Fully qualified log file name (with path).
	:return Root logger.
	"""
	logFormatter = logging.Formatter( "%(asctime)s [%(levelname)-5.5s]  %(message)s" )
	rootLogger = logging.getLogger()
	if rootLogger.hasHandlers():
		rootLogger.handlers.clear()
	rootLogger.setLevel( logging.INFO )

	# Log to file.
	fileHandler = logging.FileHandler( fileName, mode="w" )
	fileHandler.setFormatter( logFormatter )
	rootLogger.addHandler( fileHandler )

	# Log to console as well.
	consoleHandler = logging.StreamHandler( sys.stdout )
	rootLogger.addHandler( consoleHandler )

	return rootLogger


def plotCorrelation( targets: np.ndarray, predictions: np.ndarray, fileName: str="scatter1.png", fitColor: str="C1",
					 yLabel: str=r"Inferred $h\kappa^\star$", approxHK=r"h\kappa^\star", squareFlag=True, grid=False,
					 splitIdxs: Union[None, List[List[int]]]=None, splitColors: Union[None, List[str]]=None,
					 splitLabels: Union[None, List[str]]=None ):
	"""
	Plot correlation between target and predicted mean curvature.
	:param targets: Array of expected values.
	:param predictions: Array of predicted values.
	:param fileName: Where to store the figure.
	:param fitColor: What color to assign to the fit line.
	:param yLabel: Y-axis label.
	:param approxHK: Approximated hk symbol.
	:param squareFlag: Whether force square axes ratio or not.
	:param grid: Whether to enable or disable grid.
	:param splitIdxs: A list of lists of indices to split the data points into groups with different colors.
	:param splitColors: A list of colors to provide to each group of data points.
	:param splitLabels: A list of labels to assign to each group of data points.
	"""
	slope, intercept, r, _, _ = stats.linregress( targets, predictions )

	fig = plt.figure( figsize=(6,5), dpi=300 )
	ax = fig.add_subplot( 111 )
	tgtHK = r"h\kappa^*"
	plt.title( r"$\rho$ = {:.7f},   ${}$ = {:.5f}${}$ + ({:.5f})".format( r, approxHK, slope, tgtHK, intercept ) )
	if splitIdxs is None or splitColors is None or splitLabels is None or len( splitIdxs ) != len( splitColors ) or len( splitColors ) != len( splitLabels ):
		ax.scatter( targets, predictions, marker=".", facecolors="none", edgecolors="C0", label="Data" )
	else:
		for i, idxs in enumerate( splitIdxs ):
			ax.scatter( targets[idxs], predictions[idxs], marker=".", facecolors="none", edgecolors=splitColors[i], label=splitLabels[i] )
	ax.set_xlabel( r"Expected $h\kappa^*$" )
	ax.set_ylabel( yLabel )
	if squareFlag:
		ax.set_aspect( 1.0 / ax.get_data_ratio() * 1.0 )
	minClip = min( ax.get_xlim()[0], ax.get_ylim()[0] )
	maxClip = max( ax.get_xlim()[1], ax.get_ylim()[1] )
	ax.set_xlim( ax.get_xlim() )
	ax.set_ylim( ax.get_ylim() )
	minV = minClip - 1
	maxV = maxClip + 1
	ax.plot( [minV, maxV], [minV, maxV], "k--", label="${} = {}$".format( approxHK, tgtHK ) )
	ax.plot( [minV, maxV], [minV*slope + intercept, maxV*slope + intercept], color=fitColor, label="Fit line")
	ax.legend()
	plt.grid( grid )
	plt.savefig( fileName, bbox_inches="tight" )
	plt.show()


def compareErrors( targets, predictions, numerics=None, verbose=True ) -> (float, float, float, float, float, float):
	"""
	Compute and compare errors for predictions and numerical approximations.
	:param targets: Expected output values.
	:param predictions: Predicted values by the deep learning model.
	:param numerics: Approximated values by the numerical method.  If None, we don't consider numerics at all.
	:param verbose: Whether to print the comparison or not.
	:return: Various errors.
	"""
	neuralMAE = metrics.mean_absolute_error( targets, predictions )
	neuralMaxAE = max( np.abs( targets - predictions ) )
	neuralMSE = metrics.mean_squared_error( targets, predictions )
	numericsMAE = np.inf					# Change values if numerics array is given.
	numericsMaxAE = np.inf
	numericsMSE = np.inf
	if numerics is not None:
		numericsMAE = metrics.mean_absolute_error( targets, numerics )
		numericsMaxAE = max( np.abs( targets - numerics ) )
		numericsMSE = metrics.mean_squared_error( targets, numerics )
	if verbose:
		logging.info( "  Method            | Mean Absolute Error   | Max Absolute Error    | Root Mean Sq Err" )
		logging.info( "* Neural model      | {:>20.10e}  | {:>20.10e}  | {:>20.10e}".format( neuralMAE, neuralMaxAE, np.sqrt( neuralMSE ) ) )
		logging.info( "* Numerical method  | {:>20.10e}  | {:>20.10e}  | {:>20.10e}".format( numericsMAE, numericsMaxAE, np.sqrt( numericsMSE ) ) )

	return neuralMAE, neuralMaxAE, np.sqrt( neuralMSE ), numericsMAE, numericsMaxAE, np.sqrt( numericsMSE )
#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

#clear

##  MESHING ##
#------------------------------------------------------------------------------	
	rm -rf system/decomposeParDict
	cp -r system/decomposeParDict.scotch system/decomposeParDict	
	
	echo "BlockMesh"
	#cp -r dynamicCode.orig dynamicCode	
	blockMesh >> logMeshing	
#------------------------------------------------------------------------------
	#cp -r dynamicCode dynamicCode.orig
	#cp -r system/decomposeParDict.hierarchical system/decomposeParDict
#------------------------------------------------------------------------------
	#echo "decomposeParMesh"
	#decomposePar > logMeshing1

	#pwd

	#sed 's/0.0001/0.05/g' processor*/constant/polyMesh/boundary -i

	#echo "renumberMesh"
	#mpirun -np 9 renumberMesh -overwrite -parallel >> logMeshing9	

	echo "snappyHexMesh"
	snappyHexMesh -overwrite >> logMeshing
	#mpirun -np 9 snappyHexMesh -overwrite -parallel >> logMeshing

	#echo "reconstructParMesh"
	#reconstructParMesh -latestTime -mergeTol 1E-06 -noZero >> logMeshing2
	#-mergeTol 1e-06 -latestTime
#------------------------------------------------------------------------------
	echo "extrudeMesh"
	extrudeMesh >> logMeshing
	
##  SIMULATION ##
#------------------------------------------------------------------------------

	rm -rf system/decomposeParDict
	cp -r system/decomposeParDict.simple system/decomposeParDict

	echo "decomposePar"
	decomposePar > logSolve

	echo "renumberMesh"
	mpirun -np 12 renumberMesh -overwrite -parallel >> logSolve

	echo "simpleFoam"	
	mpirun -np 12 simpleFoam -parallel >> logSolve
	
	
	reconstructPar >> logsolve
    #echo "simpleFoam"
    #simpleFoam
	echo "DONE!!!"

#------------------------------------------------------------------------------
	#python ./time.py
	#echo "Postprocessing"
	#cd PythonCodes
	##python ./.py
	#cd ..

#------------------------------------------------------------------------------
##  OVER  ##
#------------------------------------------------------------------------------


# Group Problem

Please make sure the file:

IGmain.py, InfromationGain.py, VImain.py, VarianceImpurity.py and readData.py

at the same location.

Please make sure enter the right entering format.

Please make sure enter the right data location.

For compiling the code, you need to install the pandas and numpy package.

## For using Infromation Gain heuristice, please run:

python -u IGmain.py <L> <K> <training-set> <validation-set> <test-set> <to-print>

L: integer (used in the post-pruning algorithm)

K: integer (used in the post-pruning algorithm)

<training-set> <validation-set> <test-set>: data location

to-print:{yes,no}

## For using Variance impurity heuristic, please run:

python -u VImain.py <L> <K> <training-set> <validation-set> <test-set> <to-print>

L: integer (used in the post-pruning algorithm)

K: integer (used in the post-pruning algorithm)

<training-set> <validation-set> <test-set>: data location

to-print:{yes,no}
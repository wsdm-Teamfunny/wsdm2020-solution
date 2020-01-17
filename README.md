# WSDM2020-solution
## Team Name: funny
Team Member: just4fun, greedisgood, slowdown, funny
## No Data Leak
We achieve map@3 score 0.37458 at part 1 and 0.38020 at part 2 without using any data leak in the competition. During the recall process we search the related papers from the whole dataset without tricky data screening.

## Our Basic Solution 
data preprocess -> recall by text similarity-> single model (LGB + NN) -> model stacking -> linear ensemble -> final result


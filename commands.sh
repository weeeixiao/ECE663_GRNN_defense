python defense_trial.py --epochs 5 --frac 0.1 --iteration 3000 --dp_mechanism Laplace --dataset cifar10 --eval
# epoch is the epoch for FL training
# iteration is for GRNN training
# frac is the fraction of clients involved in one iter
# dp_mechanism can be set to 'no_dp' and 'Laplace'
# dataset cifar10 has been tested, others not
# --eval will print the PSNR score in stdout and on the figure

# INSTRUCTIONS:
# Current acc ploting cannot work,
# for report, I recommend using the original script like
python dpfl_main.py --dp_mechanism no_dp
# or
python dpfl_main.py --dp_mechanism Laplace
# so as to get the accuracy & loss plot for evaluation

# make device="cuda:0" run_Ising_P
# make device="cuda:0" run_Ising_M
# make device="cuda:0" run_MRF_P
# make device="cuda:0" run_MRF_M
# make device="cuda:0" run_Sex
# make device="cuda:0" run_County_Facebook
# make device="cuda:0" run_Cora
# make device="cuda:0" run_CiteSeer
# make device="cuda:0" run_PubMed
# make device="cuda:0" run_Coauthor_CS
# make device="cuda:0" run_Coauthor_Physics

# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples   5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  10 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  20 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  50 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples 100 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  -1 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device cuda:2 --learn_H --verbose

# python3 main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples   5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials 10 --device cuda:0 --loss_option 0 --learn_H --verbose


#python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  1 --device cuda:0 --learn_H --verbose --develop 

#python main.py --dataset OGBN_arXiv  --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  5 --dropout_p 0.1 --device cuda --learning_rate 1.0e-3 --num_epoches 30 --num_trials  3 --initskip_BP 0.00 --lossfunc_BP 5 --deg_scaling --learn_H --verbose --develop


#python main.py --dataset OGBN_arXiv  --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  5 --dropout_p 0.1 --device cuda --learning_rate 1.0e-3 --num_epoches 30 --num_trials  3 --initskip_BP 0.00 --lossfunc_BP 5 --deg_scaling --learn_H --verbose --develop


# Inductive
python main.py --dataset OGBN_arXiv --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples   5 --dropout_p 0.1 --device cuda --learning_rate 1.0e-3 --num_epoches 100 --num_trials  10 --initskip_BP 0.00 --lossfunc_BP 5 --deg_scaling --learn_H --verbose #--develop


# Transductive
python main.py --dataset OGBN_arXiv --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples   5 --dropout_p 0.1 --device cuda --learning_rate 1.0e-3 --num_epoches 100 --num_trials  10 --initskip_BP 0.00 --lossfunc_BP 5 --deg_scaling --learn_H --eval_C --verbose #--develop
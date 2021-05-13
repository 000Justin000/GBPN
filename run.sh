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

# make device="cuda:1" run_Elliptic_Bitcoin
# make device="cuda:1" run_JPMC_Payment0
# make device="cuda:1" run_JPMC_Payment1
# make device="cuda:1" run_arXiv
# make device="cuda:1" run_Products

# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples   5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  10 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  20 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  50 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples 100 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose
# python main.py --dataset OGBN_arXiv    --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 1 --num_samples  -1 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --loss_option 0 --num_trials  3 --device cuda:2 --learn_H --verbose

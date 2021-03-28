# make device="cuda:0" run_Sex
# make device="cuda:0" run_County_Facebook
# make device="cuda:0" run_Cora
# make device="cuda:0" run_CiteSeer
# make device="cuda:0" run_PubMed
# make device="cuda:0" run_Coauthor_CS
# make device="cuda:0" run_Coauthor_Physics
# make device="cuda:1" run_Elliptic_Bitcoin
# make device="cuda:1" run_JPMC_Fraud_Detection
# make device="cuda:1" run_arXiv
# make device="cuda:2" run_Products

python main.py --dataset OGBN_Products --model_name GAT  --dim_hidden 256 --num_layers 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials 1 --device cuda:0                    --verbose &
python main.py --dataset OGBN_Products --model_name GBPN --dim_hidden 256 --num_layers 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials 1 --device cuda:1 --learn_H --eval_C --verbose &
python main.py --dataset OGBN_Products --model_name GBPN --dim_hidden 256 --num_layers 3 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials 1 --device cuda:2 --learn_H --eval_C --verbose &

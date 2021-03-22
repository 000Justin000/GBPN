# make run_MLP
# make run_SGC
# make run_GCN
# make run_SAGE
# make run_GAT
# make run_GBPN_I
# make run_GBPN_T
# make run_Elliptic_Bitcoin
# make run_JPMC_Fraud_Detection
# make run_arXiv
# make run_Products

python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --num_epoches 200 --device cuda --num_hidden 2 --learn_H --eval_C --verbose
python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --num_epoches 200 --device cuda --num_hidden 2 --learn_H --eval_C --verbose

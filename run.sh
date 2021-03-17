# make run_SGC
# make run_GCN
# make run_SAGE
# make run_GAT
# make run_MLP
# make run_BPGNN_FFF
# make run_BPGNN_TFF
# make run_BPGNN_TFT
# make run_BPGNN_TTF
# make run_BPGNN_TTT
# make run_Animals

# python main.py --dataset OGBN_arXiv    --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP           --eval_C --verbose
python main.py --dataset OGBN_arXiv    --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H --eval_C --verbose
python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP           --eval_C --verbose
python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H --eval_C --verbose

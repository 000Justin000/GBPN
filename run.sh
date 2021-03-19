# make run_SGC
# make run_GCN
# make run_SAGE
# make run_GAT
# make run_MLP
# make run_GBPN_FFF
# make run_GBPN_TFF
# make run_GBPN_TFT
# make run_GBPN_TTF
# make run_GBPN_TTT
# make run_Animals

python main.py --dataset OGBN_arXiv    --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP           --eval_C --verbose
python main.py --dataset OGBN_arXiv    --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H          --verbose
python main.py --dataset OGBN_arXiv    --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H --eval_C --verbose
python main.py --dataset OGBN_Products --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP           --eval_C --verbose
python main.py --dataset OGBN_Products --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H          --verbose
python main.py --dataset OGBN_Products --model_name GBPN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H --eval_C --verbose

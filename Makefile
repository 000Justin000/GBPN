build_cnetworkx:
	c++ -O3 -Wall -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes) cnetworkx.cpp -o cnetworkx$(shell python3-config --extension-suffix)

run_SGC:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda

run_GCN:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda

run_SAGE:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda

run_GAT:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-3 --device cuda

run_MLP:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1

run_BPGNN_FFF:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1

run_BPGNN_TFF:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP 
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP 
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP 
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP 
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP 
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP 
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP 

run_BPGNN_TFT:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --eval_C 
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --eval_C 
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --eval_C 
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --eval_C 
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --eval_C 
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --eval_C 
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --eval_C 

run_BPGNN_TTF:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --learn_H 
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --learn_H 
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H 
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H 
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H 
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H 
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H 

run_BPGNN_TTT:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --learn_H --eval_C 
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0 --train_BP --learn_H --eval_C 
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C 
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C 
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C 
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C 
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C 

run_Products:
	python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2
	python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP
	python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --eval_C
	python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H
	python main.py --dataset OGBN_Products --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 2 --train_BP --learn_H --eval_C

run_Animals:
	python main.py --dataset Animals --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Animals --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Animals --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C
	python main.py --dataset Animals --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1 --train_BP --learn_H --eval_C

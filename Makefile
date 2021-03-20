build_cnetworkx:
	c++ -O3 -Wall -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes) cnetworkx.cpp -o cnetworkx$(shell python3-config --extension-suffix)

run_MLP:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name MLP  --learning_rate 1.0e-2 --device cuda

run_SGC:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name SGC  --learning_rate 1.0e-2 --device cuda

run_GCN:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GCN  --learning_rate 1.0e-2 --device cuda

run_SAGE:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name SAGE --learning_rate 1.0e-2 --device cuda

run_GAT:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GAT  --learning_rate 1.0e-3 --device cuda

run_GBPN_I:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H

run_GBPN_T:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C

run_arXiv:
	python main.py --dataset OGBN_arXiv --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset OGBN_arXiv --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C

run_Products:
	python main.py --dataset OGBN_Products --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H
	python main.py --dataset OGBN_Products --model_name GBPN --learning_rate 1.0e-2 --device cuda --learn_H --eval_C

run_SGC:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model SGC   --learning_rate 1.0e-2 --device cuda

run_GCN:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model GCN   --learning_rate 1.0e-2 --device cuda

run_GAT:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model GAT   --learning_rate 1.0e-3 --device cuda

run_BPGNN:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1

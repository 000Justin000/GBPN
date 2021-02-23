run_MLP:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda

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

run_BPGNN:
	python main.py --dataset Cora             --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset CiteSeer         --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 0
	python main.py --dataset PubMed           --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_CS      --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset County_Facebook  --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Sex              --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1

run_cats_dogs:
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name MLP   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name SGC   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name GCN   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name SAGE  --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name GAT   --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cats_Dogs --homo_ratio 0.0 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.1 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.2 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.3 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.4 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.5 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.6 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.7 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.8 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 0.9 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1
	python main.py --dataset Cats_Dogs --homo_ratio 1.0 --split 0.6 0.2 0.2 --model_name BPGNN --learning_rate 1.0e-2 --device cuda --num_hidden 1

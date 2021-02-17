run_cora:
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda --num_hidden 0

run_citeseer:
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda --num_hidden 0

run_pubmed:
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda

run_coauthor_cs:
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda

run_coauthor_physics:
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda

run_county_facebook:
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model SGC   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model GCN   --num_hidden 2 --learning_rate 1.0e-2 --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model GAT   --num_hidden 2 --learning_rate 1.0e-3 --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model BPGNN --num_hidden 2 --learning_rate 1.0e-2 --device cuda

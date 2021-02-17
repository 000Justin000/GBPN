run_cora:
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset Cora --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_citeseer:
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset CiteSeer --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_pubmed:
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset PubMed --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_coauthor_cs:
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset Coauthor_CS --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_coauthor_physics:
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset Coauthor_Physics --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_county_facebook:
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model SGC   --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model GCN   --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model GAT   --device cuda
	python main.py --dataset County_Facebook --split 0.6 0.2 0.2 --model BPGNN --device cuda

run_ogbn_product:
	python main.py --dataset OGBN_Products --split 0.6 0.2 0.2 --model SGC   --device cuda

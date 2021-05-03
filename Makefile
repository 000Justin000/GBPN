build_cnetworkx:
	c++ -O3 -Wall -shared -fopenmp -std=c++17 -fPIC $(shell python3 -m pybind11 --includes) cnetworkx.cpp -o cnetworkx$(shell python3-config --extension-suffix)

run_Ising_P:
	# python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset Ising+               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_Ising_M:
	# python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset Ising-               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_MRF_P:
	# python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset MRF+                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_MRF_M:
	# python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset MRF-                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_Sex:
	# python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset Sex                  --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_County_Facebook:
	# python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset County_Facebook      --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_Cora:
	# python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset Cora                 --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_CiteSeer:
	# python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	# python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device)                    --verbose
	python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H          --verbose
	python main.py --dataset CiteSeer             --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.6 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 30 --device $(device) --learn_H --eval_C --verbose

run_PubMed:
	# python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H          --verbose
	python main.py --dataset PubMed               --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H --eval_C --verbose

run_Coauthor_CS:
	# python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H          --verbose
	python main.py --dataset Coauthor_CS          --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H --eval_C --verbose

run_Coauthor_Physics:
	# python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name SGC  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name GCN  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	# python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device)                    --verbose
	python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H          --verbose
	python main.py --dataset Coauthor_Physics     --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --dropout_p 0.3 --learning_rate 1.0e-3 --num_epoches 500 --num_trials 10 --device $(device) --learn_H --eval_C --verbose

run_Elliptic_Bitcoin:
	# python main.py --dataset Elliptic_Bitcoin     --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset Elliptic_Bitcoin     --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset Elliptic_Bitcoin     --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	python main.py --dataset Elliptic_Bitcoin     --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H          --verbose
	python main.py --dataset Elliptic_Bitcoin     --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H --eval_C --verbose

run_JPMC_Payment0:
	# python main.py --dataset JPMC_Payment0        --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset JPMC_Payment0        --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset JPMC_Payment0        --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	python main.py --dataset JPMC_Payment0        --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H          --verbose
	python main.py --dataset JPMC_Payment0        --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H --eval_C --verbose

run_JPMC_Payment1:
	# python main.py --dataset JPMC_Payment1        --split 0.3 0.2 0.5 --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset JPMC_Payment1        --split 0.3 0.2 0.5 --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset JPMC_Payment1        --split 0.3 0.2 0.5 --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	python main.py --dataset JPMC_Payment1        --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H          --verbose
	python main.py --dataset JPMC_Payment1        --split 0.3 0.2 0.5 --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H --eval_C --verbose

run_arXiv:
	# python main.py --dataset OGBN_arXiv                               --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset OGBN_arXiv                               --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset OGBN_arXiv                               --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	python main.py --dataset OGBN_arXiv                               --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples 20 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H          --verbose
	python main.py --dataset OGBN_arXiv                               --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples 20 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H --eval_C --verbose

run_Products:
	# python main.py --dataset OGBN_Products                            --model_name MLP  --dim_hidden 256 --num_layers 2 --num_hops 0 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset OGBN_Products                            --model_name SAGE --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	# python main.py --dataset OGBN_Products                            --model_name GAT  --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples  5 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device)                    --verbose
	python main.py --dataset OGBN_Products                            --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples 10 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H          --verbose
	python main.py --dataset OGBN_Products                            --model_name GBPN --dim_hidden 256 --num_layers 2 --num_hops 2 --num_samples 10 --dropout_p 0.1 --learning_rate 1.0e-3 --num_epoches 100 --num_trials  3 --device $(device) --learn_H --eval_C --verbose

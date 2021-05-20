#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <array>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include <cmath>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// bool comp_first(const tuple<int,int> &x, const tuple<int,int> &y) {
//     return x.first < y.first;
// }

bool comp_first(const tuple<int,int,double> &x, const tuple<int,int,float> &y) {
    return get<0>(x) < get<0>(y);
}

struct Exp3 {
    vector<float> probability;
    vector<float> eta_;
    vector<float> sum_losses_;
    vector<float> sum_losses_sq_;
    int n_;
    double alpha_;

    void init(int n) {
        n_ = n;
        alpha_ = sqrt(log(n)*1.19);
        for (int i = 0; i < n; ++i) {
            probability.push_back(1.0 / static_cast<float>(n));
            sum_losses_.push_back(0.0);
            sum_losses_sq_.push_back(0.0);
        }
    }

    void update(vector<float> &loss) {
        // Update state based on loss
        for (int i = 0; i < n_; ++i) {
            sum_losses_[i] += loss[i];
            sum_losses_sq_[i] += pow(loss[i], 2);
        }

        // Compute etas
        for (int i = 0; i < n_; ++i) {
            eta_[i] = alpha_ / sqrt(sum_losses_sq_[i]);
        }
    }

    vector<float> get_prob() 
    {
        vector<float> weights;
        float sum_w = 0;

        for (int i = 0; i < n_; ++i) {
            float v = exp(sum_losses_[i] * eta_[i]);
            weights.push_back(v);
            sum_w += v;
        }

        auto probability = weights;

        for (int i = 0; i < n_; ++i) 
            probability[i] /= sum_w;

        return probability;
    }

};

struct Graph {
    vector<int> nodes;
    // nbrs[i]: neighbors of ith node
    // (nodeIdx, edgeIdx) tuple
    // vector<vector<tuple<int,int>>> nbrs;
    // TODO: Tuple
    vector<vector<tuple<int,int,float>>> nbrs;
    vector<Exp3> exp3s;


    Graph(int num_nodes=0) {
        for (int uid=0; uid<num_nodes; uid++)
            add_node(uid);

    }


    Graph(vector<int> input_nodes) {
        for (auto u: input_nodes)
            add_node(u);
    }

    void add_node(int u) {
        nodes.push_back(u);
        nbrs.push_back(vector<tuple<int,int,float>>());
    }

    void add_edge(int uid, int vid, int eid) {
        nbrs[uid].push_back(make_tuple(vid,eid,0));
    }

    void add_edges_from(const vector<tuple<int,int,int>> &edges) {
        for (auto edge: edges)
            add_edge(get<0>(edge), get<1>(edge), get<2>(edge));
        sort_nbrs();

        // Initialize EXP3 instances (one for each node)
        for (int i = 0; i < nbrs.size(); i++) {
            auto nbr = nbrs[i];
            auto num_neighbors = nbr.size();
            Exp3 this_exp3 = Exp3();
            this_exp3.init(num_neighbors);
            exp3s.push_back(this_exp3);
        }

        std::cout << "Done adding edges!" << endl;
    }

    int number_of_nodes(void) {
        return nodes.size();
    }

    vector<int> get_nodes(void) {
        return nodes;
    }

    vector<tuple<int,int,int>> get_edges(void) {
        vector<tuple<int,int,int>> edges;
        for (unsigned uid=0; uid<nodes.size(); uid++)
            for (auto arc: nbrs[uid])
                edges.push_back(make_tuple(uid, get<0>(arc), get<1>(arc)));
        return edges;
    }

    void sort_nbrs(void) {
        for (auto nbr: nbrs)
            sort(nbr.begin(), nbr.end(), comp_first);
    }

    void update_exps(vector<vector<double>> &log_msg) {
        // Compute loss and call update
        std::cout << "Log msg size " << log_msg.size() << endl;
        std::cout << "Exp3s size " << exp3s.size() << endl;
        for (int i = 0; i < log_msg.size(); ++i) {
            std::vector<float> loss;

            for (int j = 0; j < log_msg[i].size(); ++j) {
                float this_loss = 1.0 / exp3s[i].probability[j] * pow(log_msg[i][j], 2);
                loss.push_back(this_loss);
            }

            exp3s[i].update(loss);
        }
    }
};

void subtree_dfs(Graph& G, Graph& T, int r, int rid, int max_d, int num_samples) {
    /* G: the original graph
       T: the sampled subtree
       r: root node id in the original graph G
       rid: root node id in the combined forest TT
       max_d: maximum depth
       num_samples: number of neighbors to sample 
       p: importance sampling distrubtion over the neighbors */

    vector<tuple<int,int,int,int>> stack;
    if (max_d > 0)
        stack.push_back(make_tuple(r, rid, 0, -1));

    while (stack.size() > 0) {
        auto item = stack.back();
        stack.pop_back();
        int u = get<0>(item);
        int uid = get<1>(item);
        int d = get<2>(item);
        int p = get<3>(item);

//      cout << u << " " << uid << " " << d << " " << p << endl;
//      cout << G.nbrs[u].size() << " ";

        vector<tuple<int,int,float>> nbr;
        for (auto arc: G.nbrs[u])
            if (get<0>(arc) != p) {
                nbr.push_back(arc);
//              cout << get<0>(arc); << " ";
            }
      // cout << "Num neighbors " << nbr.size() << endl;

        vector<tuple<int,int,float>> selected_nbr;
        if ((num_samples < 0) || (nbr.size() <= num_samples))
            selected_nbr = nbr;
        else 
        {
            // sample(nbr.begin(), nbr.end(), back_inserter(selected_nbr), num_samples, mt19937{random_device{}()});

            auto prob = G.exp3s[u].probability;
            // Importance sampling of neighbors according to distribution p
            std::random_device rd;
            mt19937 gen(rd());
            discrete_distribution<> dist(prob.begin(), prob.end());

            for (int i = 0; i < num_samples; i++) 
            {
                auto index = dist(gen);
                selected_nbr.push_back(nbr[index]);
            }
            
        }

        int j = 0;
        for (auto arc: selected_nbr)
        {   
            int v = get<0>(arc);
            int vid = T.number_of_nodes();
            T.add_node(v);

            int uvid = get<1>(arc);
            int vuid = get<1>(*lower_bound(G.nbrs[v].begin(), G.nbrs[v].end(), make_tuple(u,0,0), comp_first));

            T.add_edge(uid, vid, uvid);
            T.add_edge(vid, uid, vuid);
            if (max_d > d+1)
                stack.push_back(make_tuple(v, vid, d+1, u));
//          cout << "(" << u << ", " << v << ", " << uid << ", " << vid << ")  ";
        }
//      cout << endl;
        // cout << "Done with function" << endl;
    }
}

Graph sample_subtree(Graph& G, vector<int> batch_nodes, int max_d, int num_samples) {
    unsigned batch_size = batch_nodes.size();

    vector<Graph> TList;
    for (unsigned rid=0; rid<batch_size; rid++)
    {
        Graph T = Graph();
        T.add_node(batch_nodes[rid]);
        TList.push_back(T);
    }

    // #pragma omp parallel for
    for (unsigned rid=0; rid<batch_size; rid++)
    {

        subtree_dfs(G, TList[rid], batch_nodes[rid], 0, max_d, num_samples);
        // if (((rid + 1) % 100) == 0) {
        //     cout << "Done with " << (rid + 1) << "/" << batch_size << endl;
        // }
    }

    vector<vector<int>> l2g;
    Graph TT = Graph(batch_nodes);
    int tot_count = batch_size;
    for (unsigned rid=0; rid<batch_size; rid++)
    {
        vector<int> l2g_map;
        l2g_map.push_back(rid);
        for (unsigned uid=1; uid<TList[rid].number_of_nodes(); uid++)
        {
            l2g_map.push_back(tot_count);
            TT.add_node(TList[rid].nodes[uid]);
            tot_count += 1;
        }
        l2g.push_back(l2g_map);
    }
    for (unsigned rid=0; rid<batch_size; rid++)
        for (unsigned uid=0; uid<TList[rid].number_of_nodes(); uid++)
            for (auto arc: TList[rid].nbrs[uid])
                TT.add_edge(l2g[rid][uid], l2g[rid][get<0>(arc)], get<1>(arc));

    return TT;    
}

Graph onehop_subgraph(Graph& G, vector<int> batch_nodes) {
    unsigned batch_size = batch_nodes.size();

    Graph S = Graph(batch_nodes);
    
    // map from the original indices of batch nodes to indices in the new graph S
    map<int,int> b2g_map;
    for (unsigned rid=0; rid<batch_size; rid++)
        b2g_map[batch_nodes[rid]] = rid;

    // map from the original indices of neighboring nodes to indices in the new graph S
    map<int,int> n2g_map;
    for (unsigned uid=0; uid<batch_size; uid++)
    {
        int u = batch_nodes[uid];
        for (auto arc: G.nbrs[u])
        {
            // int v = get<0>(arc);
            int v = get<0>(arc);

            int vid;

            if (n2g_map.find(v) != n2g_map.end())
            {
                vid = n2g_map[v];
            }
            else
            {
                vid = S.number_of_nodes();
                S.add_node(v);
                n2g_map[v] = vid;
            }

            // int uvid = get<1>(arc);
            int uvid = get<1>(arc);
            //int vuid = lower_bound(G.nbrs[v].begin(), G.nbrs[v].end(), make_tuple(u,0, 0), comp_first)->second;
            int vuid = get<1>(*lower_bound(G.nbrs[v].begin(), G.nbrs[v].end(), make_tuple(u,0, 0), comp_first));

            S.add_edge(uid, vid, uvid);
            S.add_edge(vid, uid, vuid);
        }
    }

    return S;
}

PYBIND11_MODULE(cnetworkx, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<int>())
        .def(py::init<vector<int>>())
        .def("add_edges_from", &Graph::add_edges_from)
        .def("number_of_nodes", &Graph::number_of_nodes)
        .def("get_nodes", &Graph::get_nodes)
        .def("get_edges", &Graph::get_edges)
        .def("update_exps", &Graph::update_exps);
    
    m.def("sample_subtree",  &sample_subtree);
    m.def("onehop_subgraph", &onehop_subgraph);
}

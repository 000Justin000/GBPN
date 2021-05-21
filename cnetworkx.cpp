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

bool comp_first(const tuple<int,int,double> &x, const tuple<int,int,double> &y) {
    return get<0>(x) < get<0>(y);
}

struct Exp3 {
    vector<double> probability_;
    vector<double> eta_;
    vector<double> sum_losses_;
    double sum_max_loss_sq_;
    int n_;
    double alpha_;

    void init(int n) {
        n_ = n;
        alpha_ = sqrt(log(n)*1.19);
        sum_max_loss_sq_ = 0;

        for (int i = 0; i < n; ++i) {
            probability_.push_back(1.0 / static_cast<double>(n));
            sum_losses_.push_back(0.0);
            eta_.push_back(1.0);
        }
    }

    void update(vector<double> &loss) {

        const double R = 10000;
        double max_sq_loss = 0.0;
        // Update state based on loss
        for (int i = 0; i < n_; ++i) {
            sum_losses_[i] += loss[i] / R;
            max_sq_loss = max(max_sq_loss, pow(loss[i] / R, 2.0));
            //cout << sum_losses_[i] << endl;
        }
        sum_max_loss_sq_ += max_sq_loss;


        // Compute etas
        for (int i = 0; i < n_; ++i) {
            eta_[i] = alpha_ / sqrt(sum_max_loss_sq_);
        }

        probability_ = get_prob();
    }

    vector<double> get_prob() 
    {
        vector<double> weights;
        double sum_w = 0.0;

        double max_loss = 0.0;

        for (int i = 0; i < n_; ++i) {
            auto v = sum_losses_[i] * eta_[i];
            if (v >= max_loss) {
                max_loss = v;
            }
        }

        for (int i = 0; i < n_; ++i) {
            double v = exp(sum_losses_[i] * eta_[i] - max_loss);
            //cout << "v[" << i << "]: " << v << endl;
            weights.push_back(v);
            sum_w += v;
        }

        auto probability = weights;

        for (int i = 0; i < n_; ++i) 
            probability[i] /= sum_w;

        cout << "(";
        for (auto prob: probability)
            cout << prob << ", ";
        cout << ")" << endl;
        return probability;
    }

};

struct Graph {
    vector<int> nodes;
    // nbrs[i]: neighbors of ith node
    // (nodeIdx, edgeIdx) tuple
    // vector<vector<tuple<int,int>>> nbrs;
    // TODO: Tuple
    vector<vector<tuple<int,int,double>>> nbrs;
    // One Exp3 instance per node.
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
        nbrs.push_back(vector<tuple<int,int,double>>());
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

    vector<double> update_exps(vector<double> &log_msg) {
        // log_msg: dimensions num_edges x num_classes
        // auto log_msg_new = log_msg.max(dim=-1) // maximum contribution/message across all labels of each edge
        // // log_msg_new has dimension num_edges

        // vector<vector<pair<int,vector<double>>> transformed_messages;
        // Compute loss and call update
        // std::cout << "Log msg size " << log_msg.size() << endl;
        // std::cout << "Exp3s size " << exp3s.size() << endl;

        // for i, (u, v) in enumerate(indices):

        //     transformed_messages[(u,v)] = log_msg[i]

        // // Sort; modify comp_first
        // srt_nbrs(transformed_messages)


        vector<double> scaling = log_msg;
        int z = 0;
        // Iterate over all nodes.
        for (int i = 0; i < nbrs.size(); ++i) {
            // Dimension: num_neighbors
            std::vector<double> loss;
            // cout << exp3s[i].probability_ << endl;
            
            // Construct loss vector by iterating over all neighbors of node i
            for (int j = 0; j < nbrs[i].size(); ++j) {
            // for (int j = 0; j < sampled_neighbors[i].size(); ++j) {
                z++;
                auto eid = get<1>(nbrs[i][j]);
                double this_loss = (1.0 / pow(exp3s[i].probability_[j], 2.0)) * pow(log_msg[eid], 2.0);
                // cout << "los_msg " << log_msg[eid] << endl;
                // cout << "this loss " << this_loss << endl;
                loss.push_back(this_loss);
                scaling[eid] = 1.0 / (exp3s[i].probability_[j]);
                // cout << "prob[i][j] " << (exp3s[i].probability_[j]) << endl;
                // cout << "Scaling[eid] " << scaling[eid] << endl;
                // cout << "actual[eid] " << 1.0 / (exp3s[i].probability_[j]) << endl;
            }

            // Update the exp3 for node i
            exp3s[i].update(loss);

            // if ((i + 1) % 100000 == 0)
            //     cout << "Updated Exp3 for node " << (i+1) << "/" << nbrs.size() << endl;
        }
        cout << "z: " << z << endl;
        cout << "log msg size: " << log_msg.size() << endl;

        return scaling;
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

        vector<tuple<int,int,double>> nbr;
        for (auto arc: G.nbrs[u])
            if (get<0>(arc) != p) {
                nbr.push_back(arc);
//              cout << get<0>(arc); << " ";
            }
      // cout << "Num neighbors " << nbr.size() << endl;

        vector<tuple<int,int,double>> selected_nbr;
        if ((num_samples < 0) || (nbr.size() <= num_samples))
            selected_nbr = nbr;
        else 
        {

            bool UNIFORM = false;
            if (UNIFORM)
                // TODO: This is not without replacement so the reweighting is actually wrong
                sample(nbr.begin(), nbr.end(), back_inserter(selected_nbr), num_samples, mt19937{random_device{}()});
            else {
                // Get the sampling distribution for the neighbors of node u.
                auto prob = G.exp3s[u].probability_;

                // Importance sampling of neighbors according to distribution p
                std::random_device rd;
                mt19937 gen(rd());
                discrete_distribution<> dist(prob.begin(), prob.end());

                // Pick num_samples weighted samples
                for (int i = 0; i < num_samples; i++) 
                {
                    auto index = dist(gen);
                    selected_nbr.push_back(nbr[index]);
                }    
            }

            

            // Update the weights of each sampled edge.
            
        }


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

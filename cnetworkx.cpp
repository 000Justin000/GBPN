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
    vector<double> last_loss_;
    double sum_max_loss_sq_;
    double C_;
    int n_;
    double alpha_;

    void init(int n) {
        n_ = n;
        alpha_ = sqrt(log(n)*1.19);
        sum_max_loss_sq_ = 0;
        C_ = 0.0;

        for (int i = 0; i < n; ++i) {
            probability_.push_back(1.0 / static_cast<double>(n));
            sum_losses_.push_back(0.0);
            last_loss_.push_back(0.0);
            eta_.push_back(1.0);
        }
    }

    void update(vector<double> &loss) {

        double max_sq_loss = 0.0;
        double max_loss = 0.0;
        bool optimistic = false;

        // Update state based on loss
        for (int i = 0; i < n_; ++i) {
            sum_losses_[i] += loss[i];

            if (optimistic) {
                max_sq_loss = max(max_sq_loss, pow(loss[i] - last_loss_[i], 2));
            } else {
                max_sq_loss = max(max_sq_loss, pow(loss[i], 2.0));
            }

            //cout << sum_losses_[i] << endl;
            C_ = max(C_, loss[i]);
        }

        sum_max_loss_sq_ += max_sq_loss;

        // Compute etas
        for (int i = 0; i < n_; ++i) {
            if (optimistic) {
                eta_[i] = alpha_ / sqrt(C_ + sum_max_loss_sq_); // optimistic version 
           } else {
                eta_[i] = alpha_ / sqrt(sum_max_loss_sq_); // non-optimistic version
           }
        }

        // Update the probability
        probability_ = get_prob();


        // Save last loss in case optimistic == true.
        for (int i = 0; i < n_; ++i) {
            last_loss_[i] = loss[i];
        }

    }

    vector<double> get_prob() 
    {
        vector<double> weights(n_, 0);
        double sum_w = 0.0;

        for (int i = 0; i < n_; ++i) {
            double v = exp((sum_losses_[i] + last_loss_[i])* eta_[i]);
            weights[i] = v;
            sum_w += v;
        }

        auto probability = weights;

        for (int i = 0; i < n_; ++i) {
            probability[i] /= sum_w;
        }


        // cout << "(";
        // for (auto prob: probability)
        //     cout << prob << ", ";
        // cout << ")" << endl;
        return probability;
    }

};


struct Graph {
    vector<int> nodes;
    vector<double> scaling_;
    // nbrs[i]: neighbors of ith node
    // (nodeIdx, edgeIdx) tuple
    // vector<vector<tuple<int,int>>> nbrs;
    // TODO: Tuple
    vector<vector<tuple<int,int,double>>> nbrs;
    // One Exp3 instance per node.
    vector<Exp3> exp3s;
    vector<double> need_scaling_update_;


    Graph(int num_nodes=0) {
        for (int uid=0; uid<num_nodes; uid++)
            add_node(uid);

    }

    Graph(vector<int> input_nodes) {
        for (auto u: input_nodes)
            add_node(u);
    }

    const vector<double> get_scaling() {
        return scaling_;
    }

    void add_node(int u) {
        nodes.push_back(u);
        nbrs.push_back(vector<tuple<int,int,double>>());
    }

    void add_edge(int uid, int vid, int eid) {
        nbrs[uid].push_back(make_tuple(vid,eid,0));
    }

    void add_edges_from(const vector<tuple<int,int,int>> &edges) {
        for (auto edge: edges) {
            add_edge(get<0>(edge), get<1>(edge), get<2>(edge));
            scaling_.push_back(1);
            need_scaling_update_.push_back(true);
        }
        sort_nbrs();

        // Initialize EXP3 instances (one for each node)
        for (int i = 0; i < nbrs.size(); i++) {
            auto nbr = nbrs[i];
            auto num_neighbors = nbr.size();
            Exp3 this_exp3 = Exp3();
            this_exp3.init(num_neighbors);
            exp3s.push_back(this_exp3);
        }

        update_scaling();
    }


    void update_scaling() {
        vector<double> scaling(scaling_.size(), 0.0);

        // Now compute the scaling
        #pragma omp parallel for
        for (int i = 0; i < nbrs.size(); ++i) {
            for (int j = 0; j < nbrs[i].size(); ++j) {
                auto eid = get<1>(nbrs[i][j]);
                scaling[eid] = 1.0 / (exp3s[i].probability_[j]); // 1.0 / p_{ij}
            }
        }

        // cout << "z: " << z << endl;
        // cout << "log msg size: " << log_msg.size() << endl;

        scaling_ = scaling;
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


    //vector<double> update_exps(vector<double> &log_msg) {
    void update_exps(vector<double> &log_msg) {

        // TODO: try loss as max_{c in classes} Var_c
        // and use subgradient.

        // vector<int> index(nbrs.size(), 0);
        // vector<double> mu_c_star(nbrs.size(), 0);
        // #pragma omp parallel for
        // for (int i = 0; i < nbrs.size(); ++i) {

        //     std::vector<double> var_per_c(log_msg[0].size(), 0.0);
        //     std::vector<double> mu_per_c(log_msg[0].size(), 0.0);
        //     for (int j = 0; j < nbrs[i].size(); ++j) {
        //         auto eid = get<1>(nbrs[i][j]);

        //         for (int k = 0; k < log_msg[eid].size(); ++k) {
        //             var_per_c[k] += pow(log_msg[eid][k], 2.0) / exp3s[i].probability_[j];
        //             mu_per_c[k] += log_msg[eid][k];
        //         }
        //     }

        //     int max_var_index = 0;
        //     double max_var = 0.0;
        //     for (int z = 0; z < var_per_c.size(); ++z) {
        //         auto v = var_per_c[z] / pow(mu_per_c[z], 2.0);
        //         if (v > max_var) {
        //             max_var = v;
        //             max_var_index = z;
        //         }
        //     }
            

        //     index[i] = max_var_index;
        //     mu_c_star[i] = mu_per_c[max_var_index];
        // }


        // Iterate over all nodes.
        #pragma omp parallel for
        for (int i = 0; i < nbrs.size(); ++i) {
            // Dimension: num_neighbors
            std::vector<double> loss(nbrs[i].size(), 0.0);
            // cout << exp3s[i].probability_ << endl;
            
            // Construct loss vector by iterating over all neighbors of node i
            for (int j = 0; j < nbrs[i].size(); ++j) {
            // for (int j = 0; j < sampled_neighbors[i].size(); ++j) {
                auto eid = get<1>(nbrs[i][j]);

                // loss = log_msg^2 / p^2 = E_{i ~ p} [ (log_msg_i/p_i)^2 ]
                double this_loss = (1.0 / pow(exp3s[i].probability_[j], 2.0)) * pow(log_msg[eid], 2.0);
                //double this_loss = (1.0 / pow(exp3s[i].probability_[j], 2.0)) * pow(log_msg[eid][index[i]], 2.0);
                //loss[j] = this_loss / pow(mu_c_star[i], 2);
                loss[j] = this_loss;
            }
            //cout << "Done with neighbor " << i << endl << endl;

            // Update the exp3 for node i
            exp3s[i].update(loss);
            need_scaling_update_[i] = true;

        }

        // All exp3s are updated, so update the scaling. (1 / p_i)
        update_scaling();
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
                // TODO: This is not without replacement so the reweighting is actually wrong. Double check this, maybe correct.
                sample(nbr.begin(), nbr.end(), back_inserter(selected_nbr), num_samples, mt19937{random_device{}()});
            else {
                // Get the sampling distribution for the neighbors of node u.
                auto prob = G.exp3s[u].probability_;

                // Importance sampling of neighbors according to distribution p
                std::random_device rd;
                mt19937 gen(rd());
                discrete_distribution<> dist(prob.begin(), prob.end());

                vector<bool> is_sampled(prob.size(), false);
                // Pick num_samples weighted samples

                // for (int i = 0; i < num_samples; i++) {
                //     auto index = dist(gen);
                //     selected_nbr.push_back(nbr[index]);
                // }    

                while (selected_nbr.size() < num_samples)
                {
                    auto index = dist(gen);
                    if (is_sampled[index] != true)
                        selected_nbr.push_back(nbr[index]);

                    is_sampled[index] = true;
                }

                // if (G.need_scaling_update_[u] == true) {
                //     for (int k = 0; k < G.nbrs[u].size(); ++k) {
                //         auto eid = get<1>(G.nbrs[u][k]);
                //         G.scaling_[eid] = 1.0 / (prob[k]); // 1.0 / p_{ij}
                //     }    
                //     G.need_scaling_update_[u] = false;
                // }
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

    // TODO: UNCOMMENT THIS FOR EXPERIMENTS
    // TODO: UNCOMMENT THIS FOR EXPERIMENTS
    // TODO: UNCOMMENT THIS FOR EXPERIMENTS
    #pragma omp parallel for
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
        .def("get_scaling", &Graph::get_scaling, py::return_value_policy::reference)
        .def("update_exps", &Graph::update_exps);
    
    m.def("sample_subtree",  &sample_subtree);
    m.def("onehop_subgraph", &onehop_subgraph);
}

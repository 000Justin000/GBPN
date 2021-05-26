#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <array>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include <cmath>
#include <limits>
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
    vector<double> theta_;
    double var_ratio_;
    double lambda_;
    double delta_;
    double sum_max_loss_sq_;
    double C_;
    int n_;
    unsigned int t_;
    bool optimistic_;
    double alpha_;
    double max_theta_;
    bool adahedge_;
    double gamma_;

    Exp3() : sum_max_loss_sq_{0}, lambda_{0},
        C_{0}, optimistic_{false}, adahedge_{false},
        t_{1}, delta_{0}, max_theta_{0} { }


    void init(int n) {
        n_ = n;
        alpha_ = log(n_ * 1.19);

        for (int i = 0; i < n; ++i) {
            probability_.push_back(1.0 / static_cast<double>(n));
            sum_losses_.push_back(0.0);
            last_loss_.push_back(0.0);
            eta_.push_back(1.0);
            theta_.push_back(0.0);
        }
    }

    // loss is n x 1 (n = number of neighbors) (l_t)
    void update_var_ratio_(vector<double> &log_msgs) {
        double variance_unif = 0;
        double variance_ours = 0;
        double variance_optimal = 0;

        //cout << "log messages (of neighbors): [";
        for (int i = 0; i < log_msgs.size(); ++i) {
            variance_unif += pow(log_msgs[i], 2) * float(log_msgs.size());
            variance_ours += pow(log_msgs[i], 2) / probability_[i];

            variance_optimal += log_msgs[i];
            //cout << log_msgs[i];
            // if (i < log_msgs.size() - 1)
            //     cout << ", ";
        }
        //cout << "]" << endl << endl;

        variance_optimal = pow(variance_optimal, 2.0);

        double ratio =  variance_unif / variance_ours;

        //double ratio = variance_unif / variance_optimal;

        //cout << "Variance reduction (var_unif / var_ours) = " << ratio << endl;
        var_ratio_ = ratio;
    }

    void update(vector<double> &loss) {

        double max_sq_loss = 0.0;
        double max_exp_term = 0.0;
        

        vector<double> g_t = loss;
        double g_t_prob_sum = 0;
        double min_g = 0;
        double inside_log = 0;

        max_theta_ = 0;

        for (int i = 0; i < n_; ++i) {
            max_exp_term = max(max_exp_term, loss[i] / lambda_ + log(probability_[i] + 1.0e-9));
        }

        // Update state based on loss
        for (int i = 0; i < n_; ++i) {
            sum_losses_[i] += loss[i];

            if (optimistic_) {
                max_sq_loss = max(max_sq_loss, pow(loss[i] - last_loss_[i], 2));
            } else {
                max_sq_loss = max(max_sq_loss, pow(loss[i], 2.0));
            }

            //cout << sum_losses_[i] << endl;
            C_ = max(C_, loss[i]);
            // gradient = -loss
            g_t[i] = (-loss[i]);
            g_t_prob_sum += g_t[i]*probability_[i];
            theta_[i] -= g_t[i];
            max_theta_ = max(theta_[i], max_theta_);
            min_g = min(min_g, g_t[i]);
            inside_log += exp(-g_t[i] / lambda_ + log(probability_[i] + 1.0e-9) - max_exp_term);
            // if (t_ > 1) {
            //     cout << "Inside log " << inside_log << endl;
            //     cout << "g_t[i] " << -g_t[i] << endl;
            //     cout << "lambda_ " << lambda_ << endl;
            // }
        }

        if (t_ == 1) {
            delta_ = g_t_prob_sum - min_g;
        } else{
            delta_ = lambda_ * (max_exp_term + log(inside_log)) + g_t_prob_sum;
        }
        // Account for error in Jensen's ineq.
        delta_ = max(delta_, 1e-9);
        // cout << "g_t_prob_sum " << g_t_prob_sum << endl;
        // cout << "min_g " << min_g << endl;
        // cout << "Delta " << delta_ << endl;

        lambda_ += 1.0 / (log(n_)) *delta_;
        sum_max_loss_sq_ += max_sq_loss;

        // Compute etas
        for (int i = 0; i < n_; ++i) {
            if (optimistic_) {
                eta_[i] = alpha_ / sqrt(2*(C_ + sum_max_loss_sq_)); // optimistic version 
           } else {
                eta_[i] = alpha_ / sqrt(sum_max_loss_sq_); // non-optimistic version
           }
        }

                // Save last loss in case optimistic_ == true.
        for (int i = 0; i < n_; ++i) {
            last_loss_[i] = loss[i];
        }

        // Update the probability
        probability_ = get_prob();


        t_++;

    }

    vector<double> get_prob() 
    {  
        if (n_ == 1)
            return probability_;

        vector<double> weights(n_, 0);
        double sum_w = 0.0;

        for (int i = 0; i < n_; ++i) {
            double v = 0;
            if (adahedge_) {
                v = exp(theta_[i] / lambda_ - max_theta_/lambda_);

            } else {
                if (optimistic_) {
                    v = exp((sum_losses_[i] + last_loss_[i])* eta_[i]);
                } else {
                    v = exp(sum_losses_[i]* eta_[i]);
                }
            }
            
            weights[i] = v;
            sum_w += v;
        }

        auto probability = weights;

        for (int i = 0; i < n_; ++i) {
            probability[i] /= sum_w;
        }

        gamma_ = 1.0 / (t_ + 1);

        for (int i = 0; i < n_; ++i) {
            probability[i] = probability[i] * (1 - gamma_) + gamma_ / n_;
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
    vector<double> var_ratios;
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
        }
        sort_nbrs();

        // Initialize EXP3 instances (one for each node)
        for (int i = 0; i < nbrs.size(); i++) {
            auto nbr = nbrs[i];
            auto num_neighbors = nbr.size();
            Exp3 this_exp3 = Exp3();
            this_exp3.init(num_neighbors);
            exp3s.push_back(this_exp3);
            var_ratios.push_back(1.0);
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

    vector<double> &get_var_ratios() {
        return var_ratios;
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

        // Divide the losses by a large number to avoid numerical imprecision and overflows
        // (Doesn't affect guarantees since we're using scale-invariant algorithms)


        const double loss_scaling = 1.0e6;
        // Iterate over all nodes.
        #pragma omp parallel for
        for (int i = 0; i < nbrs.size(); ++i) {
            // Dimension: num_neighbors
            std::vector<double> loss(nbrs[i].size(), 0.0);
            std::vector<double> this_log_msg(nbrs[i].size(), 0.0);
            // cout << exp3s[i].probability_ << endl;
            
            // Construct loss vector by iterating over all neighbors of node i
            for (int j = 0; j < nbrs[i].size(); ++j) {
            // for (int j = 0; j < sampled_neighbors[i].size(); ++j) {
                auto eid = get<1>(nbrs[i][j]);

                this_log_msg[j] = log_msg[eid];
                // loss = log_msg^2 / p^2 = E_{i ~ p} [ (log_msg_i/p_i)^2 ]
                double this_loss = (1.0 / pow(exp3s[i].probability_[j], 2.0)) * pow(log_msg[eid], 2.0);
                //double this_loss = (1.0 / pow(exp3s[i].probability_[j], 2.0)) * pow(log_msg[eid][index[i]], 2.0);
                //loss[j] = this_loss / pow(mu_c_star[i], 2);
                loss[j] = this_loss / loss_scaling;
            }
            //cout << "Done with neighbor " << i << endl << endl;

            // Update the exp3 for node i
            exp3s[i].update_var_ratio_(this_log_msg);
            exp3s[i].update(loss);
            var_ratios[i] = exp3s[i].var_ratio_;

        }

        // All exp3s are updated, so update the scaling. (1 / p_i)
        update_scaling();
    }
};
    
// imp_sampling denotes whether to use importance sampling (true) or uniform (false)
void subtree_dfs(Graph& G, Graph& T, int r, int rid, int max_d, int num_samples, bool imp_sampling) {
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

        vector<tuple<int,int,double>> selected_nbr;
        if ((num_samples < 0) || (nbr.size() <= num_samples))
            selected_nbr = nbr;
        else 
        {

            if (!imp_sampling)
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
            }
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

        }
    }
}

Graph sample_subtree(Graph& G, vector<int> batch_nodes, int max_d, int num_samples, bool imp_sampling) {
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
        subtree_dfs(G, TList[rid], batch_nodes[rid], 0, max_d, num_samples, imp_sampling);
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
        .def("get_var_ratios", &Graph::get_var_ratios, py::return_value_policy::reference)
        .def("update_exps", &Graph::update_exps);
    
    m.def("sample_subtree",  &sample_subtree);
    m.def("onehop_subgraph", &onehop_subgraph);
}

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <assert.h>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

bool comp_first(const pair<int,int> &x, const pair<int,int> &y) {
    return x.first < y.first;
}

struct Graph {
    vector<int> nodes;
    vector<vector<pair<int,int>>> nbrs;

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
        nbrs.push_back(vector<pair<int,int>>());
    }

    void add_edge(int uid, int vid, int eid) {
        nbrs[uid].push_back(make_pair(vid,eid));
    }

    void add_edges_from(const vector<tuple<int,int,int>> &edges) {
        for (auto edge: edges)
            add_edge(get<0>(edge), get<1>(edge), get<2>(edge));
        sort_nbrs();
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
                edges.push_back(make_tuple(uid, arc.first, arc.second));
        return edges;
    }

    void sort_nbrs(void) {
        for (auto nbr: nbrs)
            sort(nbr.begin(), nbr.end(), comp_first);
    }
};

void subtree_dfs(Graph& G, Graph& T, int r, int rid, int max_d, int num_samples) {
    /* G: the original graph
       T: the sampled subtree
       r: root node id in the original graph G
       rid: root node id in the combined forest TT
       max_d: maximum depth
       num_samples: number of neighbors to sample */

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

        vector<pair<int,int>> nbr;
        for (auto arc: G.nbrs[u])
            if (arc.first != p) {
                nbr.push_back(arc);
//              cout << arc.first << " ";
            }
//      cout << nbr.size() << endl;

        vector<pair<int,int>> selected_nbr;
        if ((num_samples < 0) || (nbr.size() <= num_samples))
            selected_nbr = nbr;
        else
            sample(nbr.begin(), nbr.end(), back_inserter(selected_nbr), num_samples, mt19937{random_device{}()});
            // sample(nbr.begin(), nbr.end(), back_inserter(selected_nbr), (d == 0) ? max(int(nbr.size()*0.1), num_samples) : num_samples, mt19937{random_device{}()});

        for (auto arc: selected_nbr)
        {
            int v = arc.first;
            int vid = T.number_of_nodes();
            T.add_node(v);

            int uvid = arc.second;
            int vuid = lower_bound(G.nbrs[v].begin(), G.nbrs[v].end(), make_pair(u,0), comp_first)->second;

            T.add_edge(uid, vid, uvid);
            T.add_edge(vid, uid, vuid);
            if (max_d > d+1)
                stack.push_back(make_tuple(v, vid, d+1, u));
//          cout << "(" << u << ", " << v << ", " << uid << ", " << vid << ")  ";
        }
//      cout << endl;
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

    #pragma omp parallel for
    for (unsigned rid=0; rid<batch_size; rid++)
    {
        subtree_dfs(G, TList[rid], batch_nodes[rid], 0, max_d, num_samples);
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
                TT.add_edge(l2g[rid][uid], l2g[rid][arc.first], arc.second);

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
            int v = arc.first;
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

            int uvid = arc.second;
            int vuid = lower_bound(G.nbrs[v].begin(), G.nbrs[v].end(), make_pair(u,0), comp_first)->second;

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
        .def("get_edges", &Graph::get_edges);
    
    m.def("sample_subtree",  &sample_subtree);
    m.def("onehop_subgraph", &onehop_subgraph);
}

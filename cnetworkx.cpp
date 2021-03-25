#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <assert.h>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

struct Graph {
    vector<int> nodes;
    vector<vector<tuple<int,float>>> nbrs;

    Graph(void) {
    }

    Graph(vector<int> input_nodes) {
        for (auto u: input_nodes)
            add_node(u);
    }

    void add_node(int u)
    {
        nodes.push_back(u);
        nbrs.push_back(vector<tuple<int,float>>());
    }

    void add_edge(int uid, int vid, float weight)
    {
        nbrs[uid].push_back(make_tuple(vid, weight));
    }

    void add_edges_from(const vector<tuple<int,int>> &list)
    {
        for (auto item: list)
        {
            int uid = get<0>(item);
            int vid = get<1>(item);
            add_edge(uid, vid, 1.0);
        }
    }

    void add_weighted_edges_from(const vector<tuple<int,int,float>> &list)
    {
        for (auto item: list)
        {
            int uid = get<0>(item);
            int vid = get<1>(item);
            float weight = get<2>(item);
            add_edge(uid, vid, weight);
        }
    }

    int number_of_nodes(void)
    {
        return nodes.size();
    }

    vector<int> get_nodes(void)
    {
        return nodes;
    }

    vector<tuple<int,int>> get_edges(void)
    {
        vector<tuple<int,int>> edges;
        for (unsigned uid=0; uid<nodes.size(); uid++)
            for (auto item: nbrs[uid])
                edges.push_back(make_tuple(uid, get<0>(item)));
        return edges;
    }

    vector<tuple<int,int,float>> get_weighted_edges(void)
    {
        vector<tuple<int,int,float>> weighted_edges;
        for (unsigned uid=0; uid<nodes.size(); uid++)
            for (auto item: nbrs[uid])
                weighted_edges.push_back(make_tuple(uid, get<0>(item), get<1>(item)));
        return weighted_edges;
    }
};

Graph empty_graph(unsigned num_nodes)
{
    vector<int> input_nodes;
    for (unsigned uid=0; uid<num_nodes; uid++)
        input_nodes.push_back(uid);
    return Graph(input_nodes);
}

void dfs(Graph& G, Graph& T, int r, int rid, int max_d, int num_samples)
{
    vector<tuple<int,int,int,int>> stack;
    if (max_d > 0)
        stack.push_back(make_tuple(r, rid, 0, -1));
    while (stack.size() > 0)
    {
        auto item = stack.back();
        stack.pop_back();
        int u = get<0>(item);
        int uid = get<1>(item);
        int d = get<2>(item);
        int p = get<3>(item);

//      cout << u << " " << uid << " " << d << " " << p << endl;
//      cout << G.nbrs[u].size() << " ";

        vector<tuple<int,float>> nbrs;
        for (auto item: G.nbrs[u])
            if (get<0>(item) != p)
            {
                nbrs.push_back(item);
//              cout << get<0>(item) << " ";
            }
//      cout << nbrs.size() << endl;

        vector<tuple<int,float>> selected_nbrs;
        if ((num_samples < 0) || (nbrs.size() <= num_samples))
            selected_nbrs = nbrs;
        else
            sample(nbrs.begin(), nbrs.end(), back_inserter(selected_nbrs), num_samples, mt19937{random_device{}()});
//      cout << "sampling finished" << endl;

        for (auto item: selected_nbrs)
        {
            int v = get<0>(item);
            int weight = get<1>(item);

            int vid = T.number_of_nodes();
            T.add_node(v);
//          cout << uid << " " << vid << " " << T.number_of_nodes() << " ";
            T.add_edge(uid, vid, weight);
            T.add_edge(vid, uid, weight);
            if (max_d > d+1)
                stack.push_back(make_tuple(v, vid, d+1, u));
//          cout << "(" << v << ", " << vid << ")  ";
        }
//      cout << endl;
    }
}

Graph sample_subtree(Graph& G, vector<int> batch_nodes, int max_d, int num_samples)
{
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
        dfs(G, TList[rid], batch_nodes[rid], 0, max_d, num_samples);
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
            for (auto item: TList[rid].nbrs[uid])
                TT.add_edge(l2g[rid][uid], l2g[rid][get<0>(item)], get<1>(item));

    return TT;    
}

PYBIND11_MODULE(cnetworkx, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<vector<int>>())
        .def("add_node", &Graph::add_node)
        .def("add_edge", &Graph::add_edge)
        .def("add_edges_from", &Graph::add_edges_from)
        .def("add_weighted_edges_from", &Graph::add_weighted_edges_from)
        .def("number_of_nodes", &Graph::number_of_nodes)
        .def("get_nodes", &Graph::get_nodes)
        .def("get_edges", &Graph::get_edges)
        .def("get_weighted_edges", &Graph::get_weighted_edges);
    
   m.def("empty_graph", &empty_graph);
   m.def("sample_subtree", &sample_subtree);
}

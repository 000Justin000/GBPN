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
    vector<unordered_map<int,float>> nbrws;

    Graph(vector<int> input_nodes) {
        for (auto u: input_nodes)
            add_node(u);
    }

    void add_node(int u)
    {
        nodes.push_back(u);
        nbrws.push_back(unordered_map<int,float>());
    }

    void add_edge(int uid, int vid, float weight)
    {
        nbrws[uid][vid] = weight;
    }

    void add_edges_from(const vector<tuple<int,int>> &list)
    {
        for (auto item: list)
        {
            int uid = get<0>(item);
            int vid = get<1>(item);
            nbrws[uid][vid] = 1.0;
        }
    }

    void add_weighted_edges_from(const vector<tuple<int,int,float>> &list)
    {
        for (auto item: list)
        {
            int uid = get<0>(item);
            int vid = get<1>(item);
            float weight = get<2>(item);
            nbrws[uid][vid] = weight;
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

    bool is_undirected(void)
    {
        for (unsigned i=0; i<nodes.size(); i++)
            for (auto x: nbrws[i])
                if (nbrws[x.first].find(i) == nbrws[x.first].end())
                    return false;
        return true;
    }

    vector<tuple<int,int>> get_edges(void)
    {
        vector<tuple<int,int>> edges;
        for (unsigned i=0; i<nodes.size(); i++)
            for (auto x: nbrws[i])
                edges.push_back(make_tuple(i, x.first));
        return edges;
    }

    vector<tuple<int,int,float>> get_weighted_edges(void)
    {
        vector<tuple<int,int,float>> weighted_edges;
        for (unsigned i=0; i<nodes.size(); i++)
            for (auto x: nbrws[i])
                weighted_edges.push_back(make_tuple(i, x.first, x.second));
        return weighted_edges;
    }
};

Graph empty_graph(unsigned num_nodes)
{
    vector<int> input_nodes;
    for (unsigned i=0; i<num_nodes; i++)
        input_nodes.push_back(i);
    return Graph(input_nodes);
}

void dfs(Graph& G, Graph& T, int r, int rid, int max_d, int size)
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

//      cout << G.nbrws[u].size() << endl;

        vector<int> nbrs;
        for (auto x: G.nbrws[u])
            if (x.first != p)
            {
//              cout << x.first << " ";
                nbrs.push_back(x.first);
            }
//      cout << endl;

        vector<int> selected_nbrs;
        if (nbrs.size() > 0)
            sample(nbrs.begin(), nbrs.end(), back_inserter(selected_nbrs), size, mt19937{random_device{}()});
        for (auto v: selected_nbrs)
        {
            int vid = T.number_of_nodes();
            T.add_node(v);
            T.add_edge(uid, vid, G.nbrws[u][v]);
            T.add_edge(vid, uid, G.nbrws[v][u]);
            if (max_d > d+1)
                stack.push_back(make_tuple(v, vid, d+1, u));
//          cout << "(" << v << ", " << vid << ")  ";
        }
//      cout << endl;
    }

}

Graph sample_subtree(Graph& G, vector<int> batch_nodes, int max_d, int size)
{
    assert(G.is_undirected());
    Graph T = Graph(batch_nodes);
    for (unsigned i=0; i<batch_nodes.size(); i++)
        dfs(G, T, batch_nodes[i], i, max_d, size);
    return T;
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
        .def("is_undirected", &Graph::is_undirected)
        .def("get_edges", &Graph::get_edges)
        .def("get_weighted_edges", &Graph::get_weighted_edges);
    
   m.def("empty_graph", &empty_graph);
   m.def("sample_subtree", &sample_subtree);
}

// int main()
// {
//     cout << "start" << endl;
//     Graph g;
//     cout << "graph created" << endl;
//     g.add_node();
//     g.add_node();
//     g.add_node();
//     cout << g.number_of_nodes() << endl;
//     cout << "nodes added" << endl;
//     g.add_edge(0,1);
//     g.add_edge(0,2);
// }

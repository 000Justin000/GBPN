#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

struct Graph {
    vector<int> nodes;
    vector<unordered_map<int,float>> nbrws;

    Graph(int num_nodes) {
        for (int i = 0; i < num_nodes; i++)
            add_node(i);
    }

    void add_node(int u)
    {
        nodes.push_back(u);
        nbrws.push_back(unordered_map<int,float>());
    }

    void add_edge(int uid, int vid, float weight=1.0)
    {
        nbrws[uid][vid] = weight;
        nbrws[vid][uid] = weight;
    }

    void add_edges_from(const vector<tuple<int,int>> &list)
    {
        for (auto item: list)
        {
            int uid = get<0>(item);
            int vid = get<1>(item);
            nbrws[uid][vid] = 1.0;
            nbrws[vid][uid] = 1.0;
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
            nbrws[vid][uid] = weight;
        }
    }

    int number_of_nodes(void)
    {
        return nodes.size();
    }
};

Graph empty_graph(int num_nodes)
{
    return Graph(num_nodes);
}

void dfs(const Graph& G, int r, int rid, int max_d)
{
    vector<tuple<int,int,int,int>> stack();
    if (max_d > 0)
        stack.push_back(make_tuple(r, rid, 0, -1));
    while stack.size() > 0:
    {
        auto item = stack.pop_back();
    }

}

Graph sample_subtree(const Graph& G, vector<int> batch_nodes)
{
    Graph T = empty_graph(batch_nodes.size());

}

PYBIND11_MODULE(clib, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<int>())
        .def("add_node", &Graph::add_node)
        .def("add_edge", &Graph::add_edge)
        .def("add_edges_from", &Graph::add_edges_from)
        .def("add_weighted_edges_from", &Graph::add_weighted_edges_from)
        .def("number_of_nodes", &Graph::number_of_nodes);
    
   m.def("empty_graph", &empty_graph);
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

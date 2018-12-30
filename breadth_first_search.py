
"""

Breadth-First Search.

by Eric Riddoch
Oct 26, 2018

"""

from collections import deque
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d.keys():
            self.d[n] = set()

    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        self.add_node(u)
        self.add_node(v)
        self.d[u].add(v)
        self.d[v].add(u)

    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """

        try:
            for node in self.d.keys():
                if n in self.d[node]:
                    self.d[node].remove(n)
            self.d.pop(n) # delete value from dictionary
        except KeyError as error:
            raise KeyError("Don't be dumb!")

    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """

        self.d[u].remove(v)
        self.d[v].remove(u)

    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        
        visited = [source]
        to_visit = set(self.d.keys() - {source})
        Q = deque(source)

        while len(Q) > 0:
            current = Q.popleft()

            # process the node
            for neighbor in self.d[current]:
                if neighbor in to_visit:
                    Q.append(neighbor)
                    visited.append(neighbor)
                    to_visit.remove(neighbor)
            
        return visited
        
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        
        # check if target exists in graph
        if target not in self.d.keys() or source not in self.d.keys():
            raise KeyError("Target or source is not in the graph!")

        path = []
        predecessors = {source: None}
        to_visit = set(self.d.keys() - [source])
        Q = deque(source)

        # breadth first search
        while len(Q) > 0:
            current = Q.popleft()

            # examine neighbors of current
            for neighbor in self.d[current]:
                
                # case 1 : target is found
                if neighbor == target:
                    predecessors[target] = current

                    # backtrack from target to source
                    while neighbor is not None:
                        path.append(neighbor)
                        neighbor = predecessors[neighbor]
                    return path[::-1]

                # case 2 : process neighbors
                elif neighbor in to_visit:
                    predecessors[neighbor] = current
                    Q.append(neighbor)
                    to_visit.remove(neighbor)

class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        
        # set data members
        self.graph = nx.Graph()
        self.movies = set()
        self.actors = set()

        with open(filename, encoding="utf8") as infile:
            self.contents = infile.readlines()

        # format data
        for i in range(len(self.contents)):
            self.contents[i] = self.contents[i].strip('\n').split(sep="/")

        # add edges to the graph
        for line in self.contents:
            for j in range(len(line)):
                if j == 0:
                    self.movies.add(line[j])
                else:

                    self.actors.add(line[j])
                    self.graph.add_edge(line[0], line[j])


                

    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints.
            (int): the number of steps from source to target, excluding movies.
        """
        
        # calculate shortest path
        path = nx.shortest_path(self.graph, source, target)
        return path, len(path) - len(path) // 2

    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """

        # get distances and paths to the target
        distances, paths = nx.single_source_dijkstra(self.graph, source=target)

        # a path of odd length, since there is a movie between any 2 actors
        # so, a path of length n, yields a true distance from actor to actor
        # excluding movies of n - (n // 2)

        true_distances = []

        for actor in self.actors:
            dist = distances[actor]
            true_distances.append(dist - (dist // 2))

        # plot path lengths as histogram
        bins = [i - .5 for i in range(8)]
        plt.hist(true_distances, bins)
        plt.xlabel("Path Length")
        plt.ylabel("# of Movies")
        plt.title("# of Actors to Kevin Bacon")
        plt.show()

        return np.mean(true_distances)

# example actor search
# if __name__ == "__main__":
#     graph = MovieGraph()
#     graph.path_to_actor("Robert Downey Jr.", "Kevin Bacon")

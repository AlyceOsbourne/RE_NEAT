from itertools import count
from typing import TypeVar

import genotype


class Node:
    def __init__(self, idx, layer):
        self.idx = idx
        self.layer = layer

    @classmethod
    def from_genotype(cls, node_gene):
        return cls(*node_gene)

    def to_genotype(self):
        return genotype.NodeGene(id=self.idx, type=self.layer)

    def __repr__(self):
        return f'Node(idx={self.idx}, layer={self.layer})'


class Connection:

    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    @classmethod
    def from_genotype(cls, connection_gene):
        return cls(*connection_gene)

    def to_genotype(self):
        return genotype.ConnectionGene(
            in_node=self.in_node,
            out_node=self.out_node,
            weight=self.weight,
            enabled=self.enabled,
            innovation=self.innovation
        )

    def __repr__(self):
        return f'Connection(in_node={self.in_node}, out_node={self.out_node}, weight={self.weight}, enabled={self.enabled}, innovation={self.innovation})'


class Network:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections

    @classmethod
    def from_genotype(cls, genotype):
        nodes = [Node.from_genotype(node_gene) for node_gene in genotype.nodes]
        connections = [Connection.from_genotype(connection_gene) for connection_gene in genotype.connections]
        return cls(nodes, connections)

    def to_genotype(self):
        return genotype.Genotype(
            nodes=[node.to_genotype() for node in self.nodes],
            connections=[connection.to_genotype() for connection in self.connections]
        )

    def __repr__(self):
        out = 'Network(\n'
        out += '\tnodes=[\n'
        for node in self.nodes:
            out += f'\t\t{node},\n'
        out += '\t],\n'
        out += f'\tconnections=[\n'
        for connection in self.connections:
            out += f'\t\t{connection},\n'
        out += '\t]\n)'
        return out


Population = TypeVar(
    'Population',
    bound=list[Network]
)


def default_population_creator(
        input_len,
        output_len,
        population_size,
        weight_generation_function,
        start_with_enabled_connections
) -> tuple[
    genotype.NodeIDXCounter, genotype.InnovationIDXCounter, genotype.Innovations, Population
]:
    node_idx_counter, innovation_idx_counter = count(), count()
    innovations = dict()
    base_genome = genotype.create_default_genome(
        input_len, output_len, node_idx_counter, innovation_idx_counter, innovations, weight_generation_function,
        start_with_enabled_connections
    )
    population = [
        Network.from_genotype(
            genotype.mutate(base_genome, node_idx_counter, innovation_idx_counter, innovations)) for _ in
        range(population_size)]
    return node_idx_counter, innovation_idx_counter, innovations, population


def main():
    genotype.console_build_genome_file()
    genome = genotype.from_file('x_or')
    network = Network.from_genotype(genome)
    print(network)


if __name__ == '__main__':
    main()

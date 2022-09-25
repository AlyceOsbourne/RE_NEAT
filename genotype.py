"""Linear representations of the NN, this is so we can save and load from file with ease"""
import copy
import json
import os
import random
from itertools import count
from typing import NamedTuple, Literal, TypeVar, Dict, Callable
import pprint

InnovationIDXCounter = TypeVar("InnovationIDXCounter", bound=count)
NodeIDXCounter = TypeVar("NodeIDXCounter", bound=count)
Innovations = dict[tuple[int, int]: int]
NodeType = Literal['input', 'hidden', 'output']
NodeGene = NamedTuple(
    'NodeGene', [
        ('id', int),
        ('type', NodeType)
    ])

ConnectionGene = NamedTuple(
    'ConnectionGene', [
        ('in_node', int),
        ('out_node', int),
        ('weight', float),
        ('enabled', bool),
        ('innovation', int)
    ])

Genotype = NamedTuple(
    'Genotype', [
        ('nodes', list[NodeGene]),
        ('connections', list[ConnectionGene])
    ])


def genome_to_json(genome: Genotype) -> str:
    node_mapping = {
        'input': 0,
        'hidden': 1,
        'output': 2
    }
    nodes = [(n.id, node_mapping[n.type]) for n in genome.nodes]
    connections = [(c.in_node, c.out_node, c.weight, int(c.enabled), c.innovation) for c in genome.connections]
    return json.dumps((nodes, connections))


def json_to_genome(json_str: str) -> Genotype:
    node_mapping: dict[int, NodeType] = {
        0: 'input',
        1: 'hidden',
        2: 'output'
    }
    nodes, connections = json.loads(json_str)
    nodes = [
        NodeGene(n[0], node_mapping[n[1]])
        for n
        in nodes
    ]

    connections = [
        ConnectionGene(
            c[0],
            c[1],
            c[2],
            bool(c[3]),
            c[4]
        )
        for c
        in connections
    ]
    return Genotype(nodes, connections)


def to_file(genome: Genotype, filename: str):
    os.makedirs('genomes', exist_ok=True)
    with open(f'genomes/{filename}.json', 'w') as f:
        f.write(genome_to_json(genome))


def from_file(filename: str) -> Genotype:
    with open(f'genomes/{filename}.json', 'r') as f:
        return json_to_genome(f.read())


def create_default_genome(
        input_len: int,
        output_len: int,
        node_idx_counter: NodeIDXCounter,
        innovation_idx_counter: InnovationIDXCounter,
        innovations: Innovations,
        weight_generation_function=lambda: random.uniform(-1, 1),
        start_with_enabled_connections=False
) -> Genotype:
    input_nodes = [
        NodeGene(
            next(node_idx_counter),
            'input'
        )
        for _ in range(input_len)
    ]
    output_nodes = [
        NodeGene(
            next(node_idx_counter),
            'output'
        )
        for _ in range(output_len)
    ]
    connections = []
    for input_node in input_nodes:
        for output_node in output_nodes:
            inno_id = next(innovation_idx_counter)
            innovations[(input_node.id, output_node.id)] = inno_id
            connection = ConnectionGene(
                input_node.id,
                output_node.id,
                weight_generation_function(),
                start_with_enabled_connections,
                inno_id
            )
            connections.append(connection)

    return Genotype(
        nodes=input_nodes + output_nodes,
        connections=connections
    )


def similarity(genome_a: Genotype, genome_b: Genotype):
    """Returns the similarity between two genomes out of 100%"""
    matching_genes = 0
    conns_a = {c.innovation: c for c in genome_a.connections}
    conns_b = {c.innovation: c for c in genome_b.connections}
    for matching_genes in conns_a.keys() & conns_b.keys():
        if matching_genes in conns_a and matching_genes in conns_b:
            if conns_a[matching_genes].enabled == conns_b[matching_genes].enabled:
                conn_percent = 1
            else:
                conn_percent = 0.50
            weight_percent = 1 - abs(conns_a[matching_genes].weight - conns_b[matching_genes].weight)
            matching_genes += conn_percent * weight_percent

    return matching_genes / max(len(conns_a), len(conns_b)) * 100


def crossover(genome_a: Genotype, genome_b: Genotype):
    """Returns a new genome with the connections of genome_a and genome_b"""
    genome_a_conns = {c.innovation: c for c in genome_a.connections}
    genome_b_conns = {c.innovation: c for c in genome_b.connections}
    new_connections = []
    for inno_id in genome_a_conns.keys() | genome_b_conns.keys():
        if inno_id in genome_a_conns and inno_id in genome_b_conns:
            if random.choice([True, False]):
                new_connections.append(genome_a_conns[inno_id])
            else:
                new_connections.append(genome_b_conns[inno_id])
        elif inno_id in genome_a_conns:
            new_connections.append(genome_a_conns[inno_id])
        elif inno_id in genome_b_conns:
            new_connections.append(genome_b_conns[inno_id])
    genome_a_nodes = {n.id: n for n in genome_a.nodes}
    genome_b_nodes = {n.id: n for n in genome_b.nodes}
    new_nodes = []
    for node_id in genome_a_nodes.keys() | genome_b_nodes.keys():
        if node_id in genome_a_nodes and node_id in genome_b_nodes:
            if random.choice([True, False]):
                new_nodes.append(genome_a_nodes[node_id])
            else:
                new_nodes.append(genome_b_nodes[node_id])
        elif node_id in genome_a_nodes:
            new_nodes.append(genome_a_nodes[node_id])
        elif node_id in genome_b_nodes:
            new_nodes.append(genome_b_nodes[node_id])
    return Genotype(new_nodes, new_connections)


def mutate(
        genome: Genotype,
        node_idx_counter: NodeIDXCounter,
        innovation_idx_counter: InnovationIDXCounter,
        innovations: Innovations
):
    """Mutates the given genome and returns a new one"""

    def _mutate_new_connection():
        new_connections = genome.connections.copy()
        new_nodes = genome.nodes.copy()
        input_node = random.choice(new_nodes)
        output_node = random.choice(new_nodes)
        if (input_node.id, output_node.id) in innovations:
            return Genotype(new_nodes, new_connections)
        inno_id = next(innovation_idx_counter)
        while inno_id in innovations.values():
            inno_id = next(innovation_idx_counter)
        innovations[(input_node.id, output_node.id)] = inno_id
        new_connections.append(ConnectionGene(
            input_node.id,
            output_node.id,
            random.uniform(-1, 1),
            True,
            inno_id
        ))
        return Genotype(new_nodes, new_connections)

    def _mutate_new_node():
        new_connections = genome.connections.copy()
        new_nodes = genome.nodes.copy()
        connection = random.choice(new_connections)
        if not connection.enabled:
            return Genotype(new_nodes, new_connections)
        updated_old_connection = ConnectionGene(
            connection.in_node,
            connection.out_node,
            connection.weight,
            False,
            connection.innovation
        )
        new_connections[new_connections.index(connection)] = updated_old_connection
        new_node = NodeGene(
            next(node_idx_counter),
            'hidden'
        )
        new_nodes.append(new_node)
        conn_a_inno_id = next(innovation_idx_counter)
        while conn_a_inno_id in innovations.values():
            conn_a_inno_id = next(innovation_idx_counter)
        innovations[(connection.in_node, new_node.id)] = conn_a_inno_id
        conn_b_inno_id = next(innovation_idx_counter)
        while conn_b_inno_id in innovations.values():
            conn_b_inno_id = next(innovation_idx_counter)
        innovations[(new_node.id, connection.out_node)] = conn_b_inno_id
        new_connections.append(ConnectionGene(
            connection.in_node,
            new_node.id,
            1,
            True,
            conn_a_inno_id
        ))
        new_connections.append(ConnectionGene(
            new_node.id,
            connection.out_node,
            connection.weight,
            True,
            conn_b_inno_id
        ))
        return Genotype(new_nodes, new_connections)

    def _mutate_connection_weight():
        new_connections = genome.connections.copy()
        new_nodes = genome.nodes.copy()
        connection = random.choice(new_connections)
        if not connection.enabled:
            return Genotype(new_nodes, new_connections)
        updated_connection = ConnectionGene(
            connection.in_node,
            connection.out_node,
            connection.weight + random.uniform(-0.5, 0.5),
            True,
            connection.innovation
        )
        new_connections[new_connections.index(connection)] = updated_connection
        return Genotype(new_nodes, new_connections)

    def _mutate_connection_enabled():
        new_connections = genome.connections.copy()
        new_nodes = genome.nodes.copy()
        connection = random.choice(new_connections)
        updated_connection = ConnectionGene(
            connection.in_node,
            connection.out_node,
            connection.weight,
            not connection.enabled,
            connection.innovation
        )
        new_connections[new_connections.index(connection)] = updated_connection
        return Genotype(new_nodes, new_connections)

    return random.choices(
        [
            _mutate_new_connection,
            _mutate_new_node,
            _mutate_connection_weight,
            _mutate_connection_enabled
        ],
        weights=[0.01, 0.01, 0.8, 0.08],
        k=1
    )[0]()


def fitness(genome, *measures: Callable[[Genotype], float]):
    return sum([measure(genome) for measure in measures])


def console_build_genome_file():
    f_name = input('Enter genome_name: ')
    input_len = int(input('Enter input_len: '))
    output_len = int(input('Enter output_len: '))
    node_idx_counter, innovation_idx_counter, innovations = count(), count(), dict()
    genome = create_default_genome(input_len, output_len, node_idx_counter, innovation_idx_counter, innovations)
    pprint.pprint(genome.nodes, width=1)
    pprint.pprint(genome.connections, width=1)
    if input('Save? (y/n) ') == 'y':
        to_file(genome, f_name)
    return genome


if __name__ == '__main__':
    console_build_genome_file()
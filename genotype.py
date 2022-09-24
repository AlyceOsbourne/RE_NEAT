"""Linear representations of the NN, this is so we can save and load from file with ease"""
import json
import os
import random
from itertools import count
from typing import NamedTuple, Literal, TypeVar

InnovationIDXCounter = TypeVar("InnovationIDXCounter", bound=count)
NodeIDXCounter = TypeVar("NodeIDXCounter", bound=count)
Innovations = dict[tuple[int, int]: int]

NodeGene = NamedTuple(
    'NodeGene', [
        ('id', int),
        ('type', Literal['input', 'hidden', 'output'])
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
    return json.dumps(genome._asdict())


def json_to_genome(json_str: str) -> Genotype:
    nodes, connections = json.loads(json_str).values()
    network_nodes, network_connections = [NodeGene(id=idx, type=layer) for idx, layer in nodes], [
        ConnectionGene(*args) for args in connections]
    return Genotype(nodes=network_nodes, connections=network_connections)


def to_file(genome: Genotype, filename: str):
    os.makedirs('genomes', exist_ok=True)
    with open(f'genomes/{filename}.json', 'w') as f:
        f.write(genome_to_json(genome))


def from_file(filename: str) -> Genotype:
    with open(f'genomes/{filename}.json', 'r') as f:
        return json_to_genome(f.read())


def batch_load_genomes():
    genomes = dict()
    for filename in os.listdir('genomes'):
        if filename.endswith('.json'):
            genome = from_file(filename)
            genomes[filename[:-5]] = genome
    return genomes


def batch_save_genomes(genomes: dict[str, Genotype]):
    for name, genome in genomes.items():
        to_file(genome, name)


def similarity(genome_a:Genotype, genome_b:Genotype):
    """Returns the similarity between two genomes out of 100%"""
    a_node_lookup = {
        n.id: n
        for n in genome_a.nodes
    }

    b_node_lookup = {
        n.id: n
        for n in genome_b.nodes
    }

    a_connection_lookup = {
        c.innovation: c
        for c in genome_a.connections
    }
    b_connection_lookup = {
        c.innovation: c
        for c in genome_b.connections
    }

    a_node_len = len(a_node_lookup)
    b_node_len = len(b_node_lookup)
    a_conn_len = len(a_connection_lookup)
    b_conn_len = len(b_connection_lookup)

    length_similarity = 0
    if a_node_len == b_node_len:
        length_similarity += 50
    elif a_node_len > b_node_len:
        length_similarity += 50 * (b_node_len / a_node_len)
    else:
        length_similarity += 50 * (a_node_len / b_node_len)

    if a_conn_len == b_conn_len:
        length_similarity += 50
    elif a_conn_len > b_conn_len:
        length_similarity += 50 * (b_conn_len / a_conn_len)
    else:
        length_similarity += 50 * (a_conn_len / b_conn_len)

    node_similarity = 0
    for node_id, node in a_node_lookup.items():
        if node_id in b_node_lookup and node.enabled == b_node_lookup[node_id].enabled:
            node_similarity += 100 / a_node_len
        else:
            node_similarity += 50 / a_node_len


    connection_similarity = 0
    for conn_id, conn in a_connection_lookup.items():
        if conn_id in b_connection_lookup:
            connection_similarity += 100 / a_conn_len
        else:
            connection_similarity += 50 / a_conn_len

    return (length_similarity + node_similarity + connection_similarity) / 3


def create_default_genome(
        input_len:int,
        output_len:int,
        node_idx_counter: NodeIDXCounter,
        innovation_idx_counter: InnovationIDXCounter,
        innovations: Innovations):
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
                1,
                True,
                inno_id
            )
            connections.append(connection)

    return Genotype(
        nodes=input_nodes + output_nodes,
        connections=connections
    )


def main():
    node_idx_counter, innovation_idx_counter, innovations = count(), count(), dict()
    genome = create_default_genome(2, 1, node_idx_counter, innovation_idx_counter, innovations)
    print(genome, genome_to_json(genome), sep='\n')
    to_file(genome, 'x_or')


if __name__ == '__main__':
    main()


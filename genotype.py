"""Linear representations of the NN, this is so we can save and load from file with ease"""
import json
import os
import random
from itertools import count
from typing import NamedTuple, Literal

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


def create_default_genome(input_len: int, output_len: int) -> tuple[
        Genotype,
        count,
        dict[tuple[int, int], int],
        count
]:
    # we want to return a genome where all input nodes are connected to all output nodes
    # with no hidden nodes.
    # we also want to assign innovation numbers to each connection gene, storing in a dict to be returned also
    # we also want to assign node ids to each node gene, storing in a set to be returned also

    nodes = []
    connections = []
    innovations = dict()
    innovation_idx = count()
    node_idx = count()

    for i in range(input_len):
        nodes.append(NodeGene(id=next(node_idx), type='input'))
    for j in range(input_len, input_len + output_len):
        nodes.append(NodeGene(id=next(node_idx), type='output'))
    for i in range(input_len):
        for j in range(input_len, input_len + output_len):
            innovation = next(innovation_idx)
            innovations[(i, j)] = innovation
            connections.append(ConnectionGene(in_node=i, out_node=j, weight=random.random(), enabled=True, innovation=innovation))

    return (
        Genotype(
            nodes,
            connections),
        innovation_idx,
        innovations,
        node_idx
    )

import os
from itertools import count

import genotype, phenotype, activation_functions

x_or_settings = {
    'inputs': 2,
    'outputs': 1,
    'file_name': 'x_or',
    'hidden_node_activation': 'sigmoid',
    'output_node_activation': 'sigmoid',
}

os.makedirs('genomes', exist_ok=True)

try:
    genome = genotype.from_file(x_or_settings['file_name'])
    innovation_idx = count()
    innovations = dict()
    node_idx = count()
    for node in genome.nodes:
        next(node_idx)
    for connection in genome.connections:
        innovations[(connection.in_node, connection.out_node)] = connection.innovation
        next(innovation_idx)
    print('Loaded genome from file')
except FileNotFoundError:
    print('No genome found, creating new genome')
    genome, innovation_idx, innovations, node_idx = genotype.create_default_genome(
        x_or_settings['inputs'], x_or_settings['outputs'])
    genotype.to_file(genome, x_or_settings['file_name'])


print('Genotype:')
for node in genome.nodes:
    print(f'\t{node}')
for connection in genome.connections:
    print(f'\t{connection}')
network = phenotype.Network.from_genotype(
    genome
)
print()
print('Resulting Phenotype:')
print(*[f"\t{line}" for line in str(network).splitlines()], sep='\n')
print(innovations)
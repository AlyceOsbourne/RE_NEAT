import pprint
import random

import genotype, phenotype


class Session:
    node_idx_counter: genotype.NodeIDXCounter
    innovation_idx_counter: genotype.InnovationIDXCounter
    innovations: genotype.Innovations
    population: phenotype.Population

    def __init__(
            self,
            node_idx_counter: genotype.NodeIDXCounter,
            innovation_idx_counter: genotype.InnovationIDXCounter,
            innovations: genotype.Innovations,
            population: phenotype.Population
    ):
        self.node_idx_counter = node_idx_counter
        self.innovation_idx_counter = innovation_idx_counter
        self.innovations = innovations
        self.population = population

    @classmethod
    def create_default_session(cls, input_len, output_len, population_size, weight_generation_function,
                               start_with_enabled_connections):
        return cls(
            *phenotype.default_population_creator(
                input_len,
                output_len,
                population_size,
                weight_generation_function,
                start_with_enabled_connections
            )
        )

    def __str__(self):
        out = 'Session(\n'
        out += pprint.pformat(self.population, indent=4)
        out += '\n)'
        return out


if __name__ == '__main__':
    session = Session.create_default_session(2, 1, 100, lambda: random.uniform(-1, 1), True)
    print(session)

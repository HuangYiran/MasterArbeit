#-*- coding: UTF-8 -*-
import numpy as np
import random
import sys
sys.path.append('')

class Genetic:
    def __init__(self):
        self.P = 100 # population size
        self.I = 18*2 + 18* 10 # string length: 2 bits for control params 10 bit for weight params 
        self.pc = 0.6 # probability of performing corssover
        self.pm = 0.05 # probability of mutation
        self.ec = 0.55
        self.epoch = 10000 # max round to update the population
        self.population = Population(self.P)

    def run(self):
        for i in range(self.epoch):
            self.population.cross_over(self.pc) # with ranking
            self.population.mutation(self.pm)
            self.population.sort()
            if self.population.get_best() > self.ec:
                break
            else:
                print self.population.get_best()

class Population:
    def __init__(self, P = 1000):
        self.P = P
        self.population = []
        self._generate_population()

    def cross_over(self, pc):
        num_choose = int(self.P*pc)
        if num_choose % 2 == 1:
            num_choose -= 1
        chosen_indices = random.sample(range(self.P), num_choose)
        chosen = []
        for i in chosen_indics:
            chosen.append(self.population.pop(i))
        rest = self.population
        newborn = []
        indices = random.shuffle(range(num_choose))
        for i in range(num_choose/2):
            chrom1 = chosen[indices[i]]
            chrmo2 = chosen[indices[i+1]]
            new1, new2 = chrom1.cross_with(chrom2)
            newborn.append(new1)
            newborn.append(new2)
        self.population.extend(newborn)

    def get_best(self):
        return self.population[0].get_value()

    def mutation(self, pm):
        num_choose = int(self.P*pm)
        chosen_indices = random.sample(range(self.P), num_choose)
        chosen = []
        for i in chosen_indics:
            chosen.append(self.population.pop(i))
        mutate = []
        for i in chosen:
            mutate.append(i.mutation)
        self.population.extend(mutate)

    def sort(self):
        sequence = [li.get_fitness() for li in population]
        self.sequence = list(zip(*sorted(zip(self.population, sequence) ,key = lambda x: x[-1], reverse = True)))[0]

    def _generate_population(self):
        for i in range(P):
            chrom = Chromosome()
            self.population.append(Chrom)

class Chromosome:
    def __init__(self, num_genes = 18, random = True):
        self.genes = {}
        self.num_genes = num_genes
        self._generate_chromosome(random)
        self.fitness = self.calcu_fitness()

    def cross_with(self, chrom2):
        side1 = random.sample(range(self.num_genes), int(self.num_genes/2))
        new1 = {}
        new2 = {}
        for i in range(self.num_genes):
            key_father = self.genes[i].get_key()
            key_mother = chrom2.genes[i].get_key()
            value_father = self.genes[i].get_value()
            value_mother = chrom2.genes[i].get_value()
            if i in side1:
                key_new1 = key_father
                value_new1 = value_father
                key_new2 = key_mother
                value_new2 = value_mother
            else:
                key_new1 = key_mother
                value_new1 = value_mother
                key_new2 = key_father
                value_new2 = value_father
            new1[i] = Gene(key_new1, value_new1)
            new2[i] = Gene(key_new2, value_new2)
        new1.calcu_fitness()
        new2.calcu_fitness()
        return new1, new2

    def get_fitness():
        return self.fitness

    def mutation(self, num_mutate):
        indices = random.sample(range(self.num_genes), num_mutate)
        mutate = {}
        for i in range(self.num_genes):
            if i not in indices:
                continue
            key = True if random.randint(0,1) == 1 else False
            value = random.randint(0, 100)
            mutate.genes[i] = Gene(key, value)
        mutate.calcu_fitness()
        return mutate

    def calcu_fitness():
        # specific
        scores = '/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores'
        tgt = [int(li.rstrip('\n')) for li in open(scores)]
        root = '/tmp/decMixture_2016_'
        bs, nd = np.load(root+'s1_0.npy').shape
        s1 = np.zeros([bs, nd])
        s2 = np.zeros([bs, nd])
        ref = np.zeros([bs, nd])
        for i in range(self.num_genes):
            if self.genes[i].get_key():
                ratio = self.genes[i].get_value * 0.01
                s1 += ratio * np.load(root+'s1_'+str(i)+'.npy')
                s2 += ratio * np.load(root+'s2_'+str(i)+'.npy')
                ref += ratio * np.load(root+'ref_'+str(i)+'.npy')
        d1 = [np.linalg.norm(l1 - l2, ord = 1) for l1, l2 in zip(s1, ref)]
        d2 = [np.linalg.norm(l1 - l2, ord = 1) for l1, l2 in zip(s2, ref)]
        c = [self._compare(l1, l2) for l1, l2 in zip(d1, d2)]
        taul = valTauLike(tgt, c)
        return taul

    def _compare(self, a, b):
        if a>b:
            return 1
        elif a<b:
            return -1
        else:
            return 0

    def _generate_chromosome(self, random):
        if random:
            for i in range(num_genes):
                key = True if random.randint(0,1) == 1 else False
                value = random.randint(0, 100)
                self.genes[i] = Gene(key, value)
        else:
            for i in range(num_genes):
                if i == 6:
                    key = 1
                else:
                    key = 0
                value = 100
                self.genes[i] = Genes(key, value)


class Gene:
    def __init__(self, key, value):
        assert(type(key) == bool and type(value) == int and value >= 0 and value <= 100)
        self.key = key
        self.value = vlaue

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def get_gene(self):
        return self.key, self.value

    def set_key(self, key):
        self.key = key

    def set_value(self, value):
        self.value = value

    def set_gene(self, key, value):
        self.key = key
        self.value = value

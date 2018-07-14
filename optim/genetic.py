#-*- coding: UTF-8 -*-
import numpy as np
import random
import sys
sys.path.append('./utils')

from valuation import valTauLike

class Genetic:
    def __init__(self):
        self.P = 100 # population size
        self.I = 18*2 + 18* 10 # string length: 2 bits for control params 10 bit for weight params 
        self.pc = 0.8 # probability of performing corssover
        self.pm = 0.05 # probability of mutation
        self.ec = 0.55
        self.epoch = 10000 # max round to update the population
        print('>>> generate population')
        self.population = Population(self.P)

    def run(self):
        for i in range(self.epoch):
            print('#'*50)
            print ('round {}'.format(i))
            print('#'*50)
            print('>>> choose')
            self.population.choose()
            print('>>> cross over')
            self.population.cross_over(self.pc) # with ranking
            print('>>> mutation')
            self.population.mutation(self.pm)
            print('>>> sort with fitness')
            self.population.sort()
            genes, fitness = self.population.get_best()
            np.save('/tmp/genetic_'+str(i), genes)
            if fitness > self.ec:
                print fitness
                break
            else:
                print fitness

class Population:
    def __init__(self, P = 1000):
        self.P = P
        self.population = []
        self._generate_population()
        step = 80*1./self.P
        self.thresh = [100-i*step for i in range(self.P)]
        self.sort()

    def choose(self):
        chosen = [self.population[i] for i in range(self.P) if random.randint(0, 100) <= self.thresh[i]]
        self.population = chosen
        print('population size after chosen: {}'.format(len(self.population)))

    def cross_over(self, pc):
        num_choose = int(len(self.population)*pc)
        if num_choose % 2 == 1:
            num_choose -= 1
        chosen_indices = random.sample(range(len(self.population)), num_choose)
        chosen = [self.population[i] for i in chosen_indices]
        left = [self.population[i] for i in range(len(self.population)) if i not in chosen_indices]
        newborn = []
        indices = range(num_choose)
        random.shuffle(indices)
        for i in range(num_choose/2):
            it = indices[i]
            ch = chosen[it]
            chrom1 = chosen[indices[i]]
            chrom2 = chosen[indices[i+1]]
            new1, new2 = chrom1.cross_with(chrom2)
            newborn.append(new1)
            newborn.append(new2)
        if len(left) == 0:
            new_generation = newborn
        else:
            new_generation = left
            new_generation.extend(newborn)
        num_lack = self.P - len(new_generation)
        if num_lack >= 1:
            if len(chosen)<num_lack:
                new_generation.extend(chosen)
                for i in range(num_lack-len(chosen)):
                    chrom = Chromosome()
                    new_generation.append(chrom)
            else:
                new_generation.extend(random.sample(chosen, num_lack))
        self.population = new_generation
        print('num of newborn: {}'.format(num_choose))

    def get_best(self):
        return self.population[0].genes, self.population[0].get_fitness()

    def mutation(self, pm):
        num_choose = int(self.P*pm)
        chosen_indices = random.sample(range(self.P), num_choose)
        chosen = []
        for i in chosen_indices:
            self.population[i].mutation()
        print('num of mutation: {}'.format(num_choose))

    def sort(self):
        sequence = [li.get_fitness() for li in self.population]
        self.sequence = list(zip(*sorted(zip(self.population, sequence) ,key = lambda x: x[-1], reverse = True)))[0]

    def _generate_population(self):
        for i in range(self.P):
            chrom = Chromosome()
            self.population.append(chrom)

class Chromosome:
    def __init__(self, genes = None, num_genes = 18, random = True):
        if genes:
            self.genes = genes
            self.num_genes = len(self.genes.keys())
        else:
            self.genes = {}
            self.num_genes = num_genes
            self._generate_chromosome(random)
        self.fitness = self._calcu_fitness()

    def cross_with2(self, chrom2):
        """
        !!!!!Suspent, the kind of the excellent parents may be bad in the next generation.
        """
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
        new1 = Chromosome(new1)
        new2 = Chromosome(new2)
        return new1, new2

    def cross_with(self, chrom2):
        """
        keep the key and change the value randomly, only mutation can change the key 
        """
        side1 = random.sample(range(self.num_genes), int(self.num_genes/2))
        new1 = {}
        new2 = {}
        for i in range(self.num_genes):
            key_father = self.genes[i].get_key()
            key_mother = chrom2.genes[i].get_key()
            value_father = self.genes[i].get_value()
            value_mother = chrom2.genes[i].get_value()
            key_new1 = key_father
            key_new2 = key_mother
            value_new1 = value_father + random.randint(-10, 10)
            value_new2 = value_mother + random.randint(-10, 10)
            if value_new1 > 100:
                value_new1 = 100
            elif value_new1 < 0:
                value_new1 = 0
            if value_new2 > 100:
                value_new2 = 100
            elif value_new2 < 0:
                value_new2 = 0
            new1[i] = Gene(key_new1, value_new1)
            new2[i] = Gene(key_new2, value_new2)
        new1 = Chromosome(new1)
        new2 = Chromosome(new2)
        return new1, new2

    def get_fitness(self):
        return self.fitness

    def mutation(self, num_mutate = 3):
        indices = random.sample(range(self.num_genes), num_mutate)
        for i in range(self.num_genes):
            if i not in indices:
                continue
            key = True if random.randint(0,1) == 1 else False
            value = random.randint(0, 100)
            self.genes[i] = Gene(key, value)
        self._calcu_fitness()

    def _calcu_fitness(self):
        # specific
        scores = '/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores'
        tgt = [int(li.rstrip('\n')) for li in open(scores)]
        root = '/tmp/decMixture_2016_'
        bs, nd = np.load(root+'s1_0.npy').shape
        s1 = np.zeros([bs, nd])
        s2 = np.zeros([bs, nd])
        ref = np.zeros([bs, nd])
        values = [self.genes[i].get_value() for i in range(self.num_genes) if self.genes[i].get_key()]
        s = sum(values)
        for i in range(self.num_genes):
            if self.genes[i].get_key():
                ratio = self.genes[i].get_value()
                s1 += (ratio*1./s) * np.load(root+'s1_'+str(i)+'.npy')
                s2 += (ratio*1./s) * np.load(root+'s2_'+str(i)+'.npy')
                ref += (ratio*1./s) * np.load(root+'ref_'+str(i)+'.npy')
        d1 = [np.linalg.norm(l1 - l2, ord = 1) for l1, l2 in zip(s1, ref)]
        d2 = [np.linalg.norm(l1 - l2, ord = 1) for l1, l2 in zip(s2, ref)]
        c = [self._compare(l1, l2) for l1, l2 in zip(d1, d2)]
        taul = valTauLike(tgt, c)
        #print(taul)
        return taul

    def _compare(self, a, b):
        if a>b:
            return 1
        elif a<b:
            return -1
        else:
            return 0

    def _generate_chromosome(self, rand):
        if rand:
            for i in range(self.num_genes):
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
        self.value = value

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

import json
import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, List
from itertools import combinations
from scipy.spatial.distance import cdist


import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]


class Map:
    """
    Class to represent a map of any dimensionality, backed by a numpy array.

    This class is necessary to handle the circular buffer for the history dimension.
    """

    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        """
        Class to represent a map of any dimensionality, backed by a numpy array.

        This class is a wrapper around a numpy array that handles the circular
        buffer for the history dimension. We use it for the array of fitnesses,
        genomes, and to track whether a niche in the map has been explored or not.

        Args:
            dims (tuple): Tuple of ints representing the dimensions of the map.
            fill_value (float): Fill value to initialize the array.
            dtype (type, optional): Type to pass to the numpy array to initialize
                it. For example, we initialize the map of genomes with type `object`.
                Defaults to np.float32.
            history_length (int, optional): Length of history to store for each
                niche (cell) in the map. This acts as a circular buffer, so after
                storing `history_length` items, the buffer starts overwriting the
                oldest items. Defaults to 1.
        """
        # need to remove buffer + array replace by list of dic or object phenotypes?
        self.history_length: int = history_length #useless now
        self.dims: tuple = dims
        self.archive = []
        # if self.history_length == 1:
        #     self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        # else:
        #     # Set starting top of buffer to 0 (% operator)
        #     self.top = np.full(dims, self.history_length - 1, dtype=int) # array pointer 
        #     self.array = np.full(
        #         (history_length,) + tuple(dims), fill_value, dtype=dtype
        #     ) # array to save all puzzles 

    def __getitem__(self, map_ix):
        """If history length > 1, the history dim is an n-dim circular buffer."""
        return self.archive[map_ix]
      
    def __getrandomitem__(self):
        if self.history_length == 1:
            return "" 
        else:
            return ""

    @property
    def empty(self):
        return len(self.archive) == 0

    def __setitem__(self, map_ix, value):
        # need to just append item to archive 
        self.archive.append(value)

    def assign_fitness_in_depth(self, map_ix, value: float) -> int:
        # never used what is it for?
        raise NotImplementedError
        indices_at_bin = (slice(None),) + map_ix
        # expecting a non-empty index, only calling this method when we know
        # current fitness can be placed somewhere
        insert_idx = np.where(self.array[indices_at_bin] < value)[0][-1]
        new_bin_fitnesses = np.concatenate(
            (
                self.array[indices_at_bin][1 : insert_idx + 1],
                np.array([value]),
                self.array[indices_at_bin][insert_idx + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_fitnesses
        return insert_idx

    def insert_individual_at_depth(self, map_ix, depth, individual):
        raise NotImplementedError
        indices_at_bin = (slice(None),) + map_ix
        new_bin_individuals = np.concatenate(
            (
                self.array[indices_at_bin][1 : depth + 1],
                np.array([individual]),
                self.array[indices_at_bin][depth + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_individuals

    @property
    def latest(self) -> np.ndarray:
        """Returns the latest values in the history buffer."""
        return self.archive


    @property
    def shape(self) -> tuple:
        """Wrapper around the shape of the numpy array."""
        # doesn't make sense now
        return len(self.archive)

    @property
    def map_size(self) -> int:
        """Returns the product of the dimension sizes, not including history."""
        return len(self.archive)
            # if self.history_length == 1:
            #     return self.array.size
            # else:
            #     return self.array[0].size

    @property
    def qd_score(self) -> float:
        """Returns the quality-diversity score of the map."""
        return self.fitnesses.sum()

    @property
    def max(self) -> float:
        """Returns the maximum value in the map, not including history."""
        return np.max(self.fitnesses)

    @property
    def min(self) -> float:
        """Returns the minimum value in the map, not including history."""
        return np.min(self.fitnesses)

    @property
    def max_finite(self) -> float:
        """Returns the maximum finite value in the map, not including history."""
        if not np.isfinite(self.fitnesses).any():
            return np.NaN
        else:
            return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).max()

    @property
    def min_finite(self) -> float:
        """Returns the minimum finite value in the map, not including history."""
        if not np.isfinite(self.fitnesses).any():
            return np.NaN
        else:
            return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).min()

    @property
    def mean(self) -> float:
        """Returns the mean finite value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).mean()

    @property
    def niches_filled(self) -> int:
        """Returns the number of niches in the map that have been explored."""
        return len(self.nonzero.keys())
        # if self.history_length>1:
        #     n_niches = self.array.shape[-1]
        #     non_empty_niches = 0

        #     for niche_idx in range(n_niches):
        #         if not np.all(np.isinf(self.array[:, niche_idx])):
        #             non_empty_niches += 1
        #     return non_empty_niches
        # else:
        #     return np.count_nonzero(np.isfinite(self.array))


class MAPElitesBase:
    """
    Base class for MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        init_map: Optional[Map] = None,
    ):
        """
        The base class for MAP-Elites and variants, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_map (Map, optional): A map to use for the algorithm. If not passed,
            a new map will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.history_length = self.config.history_length
        self.save_history = self.config.save_history
        self.save_all_individual = self.config.save_all_individual
        
        if self.save_all_individual:
            self.list_of_all_individuals = []
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        np.random.seed(self.config.seed)  
        self.rng_generators = None

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        self._init_discretization()
        self._init_maps(init_map, self.config.log_snapshot_dir)
        # print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self):
        """Initializes the discretization of the behavior space."""
        raise NotImplementedError

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        raise NotImplementedError

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def visualize(self):
        """Visualizes the map."""
        pass

    def _init_maps(
        self, init_map: Optional[Map] = None, log_snapshot_dir: Optional[str] = None
    ):
        # perfomance of niches
        if init_map is None:
            self.map_dims = self._get_map_dimensions()

            #check fitnesses

            self.fitnesses =[]
        else:
            self.map_dims = init_map.dims
            self.fitnesses = init_map

        # niches' sources
        self.genomes: Map = Map(
            dims=self.map_dims,
            fill_value=0.0,
            dtype=object,
            history_length=self.history_length,
        )
        # index over explored niches to select from
        self.nonzero = {}

        log_path = Path(log_snapshot_dir)
        
        if self.config.loading_snapshot_map and os.path.isdir(log_path):
            print("loading path")
            stem_dir = log_path.stem

            assert (
                "step_" in stem_dir
            ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
            self.start_step = (
                int(stem_dir.replace("step_", "")) + 1
            )  # add 1 to correct the iteration steps to run

            with open(log_path / "config.json") as f:
                old_config = json.load(f)

            snapshot_path = log_path / "maps.pkl"
            assert os.path.isfile(
                snapshot_path
            ), f'{log_path} does not contain map snapshot "maps.pkl"'
            # first, load arrays and set them in Maps
            # Load maps from pickle file
            with open(snapshot_path, "rb") as f:
                maps = pickle.load(f)


            self.genomes = maps["genomes"]
            self.fitnesses = maps["fitnesses"]
            self.nonzero = maps["nonzero"] #self.nonzero.array = maps["nonzero"]
            # check if one of the solutions in the snapshot contains the expected genotype type for the run
            assert  len(self.nonzero.keys())!=0, "snapshot to load contains empty map"

            assert (
                self.env.config.env_name == old_config["env_name"]
            ), f'unmatching environments, got {self.env.config.env_name} and {old_config["env_name"]}'



            history_path = log_path / "history.pkl"
            if self.save_history and os.path.isfile(history_path):
                with open(history_path, "rb") as f:
                    self.history = pickle.load(f)
            with open((log_path / "fitness_history.pkl"), "rb") as f:
                self.fitness_history = pickle.load(f)

            if self.load_np_rng_state:
                with open((log_path / "np_rng_state.pkl"), "rb") as f:
                    self.rng_generators = pickle.load(f)
                    self.rng = self.rng_generators["qd_rng"]
                    self.env.set_rng_state(self.rng_generators["env_rng"])

            print("Loading finished")
            
    def __getallitems__(self) -> List[Phenotype]:
        """
        Returns all the phenotypes that are in the Map."""
        return self.genomes.archive
    
    def rd_niche_selection_explored(self,n=1) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        if len(self.nonzero.keys()) == 0:
            raise ValueError('Empty map')
        
        list_idx_sampled = self.rng.choice(len(self.nonzero.keys()),size=n) 
        list_niche_idx =  [list(self.nonzero.keys())[idx_sampled] for idx_sampled in list_idx_sampled]
        return list_niche_idx
    
    def sample_examples_from_niche(self,niche_idx) -> int:
        """Sample one example given a niche"""
        size_niche = len(self.nonzero[niche_idx])
        match self.config.sampling_strategy:
            case 'uniform':
                # sample a random niche
                # print(f'nonzero {self.nonzero}')
                
                if size_niche == 0:
                    raise ValueError('Empty niche')
                archive_index = self.rng.choice(self.nonzero[niche_idx]) # sample a random individual

            case 'prob_best_5':
                # sample a random niche and a random individual in a niche
                idx_sampled = self.rng.choice(len(self.nonzero.keys()))
                niche_idx = list(self.nonzero.keys())[idx_sampled]
                # sort_keys = sorted(lisself.nonzero.keys())
                fitness_range = [self.min_fitness(), self.max_fitness()]  # can these be -inf/+inf?
                # sort indices by fitness
                fit_idx = [(idx, self.fitnesses[idx]) for idx in self.nonzero[niche_idx]]
                print(f'fitnesses {[f for _, f in fit_idx]}')
                print(f'fitness range {fitness_range}')
                fit_idx = sorted(fit_idx, key=lambda x: x[1])[::-1][:5]  # 5 most fit
                if fitness_range[1] - fitness_range[0] == 0:
                    L = 1.
                else:
                    L = fitness_range[1] - fitness_range[0]
                normalized_fitnesses = [(f - fitness_range[0]) / L for _, f in fit_idx]
                normalized_fitnesses = np.array(normalized_fitnesses)
                if normalized_fitnesses.sum() == 0:  # all the individuals have the lowest possible fitness
                    normalized_fitnesses = np.ones_like(normalized_fitnesses) / len(normalized_fitnesses)
                else:
                    normalized_fitnesses = normalized_fitnesses / normalized_fitnesses.sum()
                print(f'probabilities {normalized_fitnesses}')
                archive_index = self.rng.choice([idx for idx, f, in fit_idx], p=normalized_fitnesses)
                
            case _:
                raise NotImplementedError(f'Unrecognized sampling strategy "{self.config.sampling_strategy}"')
        archive_index = int(archive_index)
        return archive_index
    
    def random_selection(self,trainset_only=False) -> MapIndex:
        """select a individual from a random niche (cell) in the map that has been explored.
        trainset_only = True -> don't use archive to select examples just p3 trainset"""
        #sample a niche
        (list_few_shot_example_phenotypes, skill_targeted) = self.sample_examples(trainset_only)
            
        return (list_few_shot_example_phenotypes, skill_targeted)


    def skill_sampling(self,mode=None):
        if mode is None:
            mode = self.env.config.IMGEP_mode
        n_skills = self.env.config.n_descriptor
        max_descriptor_targeted = self.env.config.max_descriptor_targeted
        skills = list(range(1, n_skills+1)) # need to put n_skills in confige

        # Generate all combinations of up to 5 skills
        skill_combinations = set()
        for r in range(1, max_descriptor_targeted+1):  # From 1 skill to 5 skills
            skill_combinations.update(combinations(skills, r))
        skill_combinations = list(skill_combinations)
        match mode:
            case 'uniform':
                idx = self.rng.choice(len(skill_combinations))
                out = skill_combinations[idx]
                skill_targeted = [1 if i in out else 0 for i in range(n_skills)]
            case 'smart':
                
                all_emb = list(self.nonzero.keys())
                all_emb = np.array([list(eval(i)) for i in all_emb]) # list of all explored niches
                
                skill_combinations_bin = [[1 if i in vec else 0 for i in range(n_skills)] for vec in skill_combinations] #list of all possible niches 
                
                #compute distance between all possible niche and all explored niches
                out=cdist(skill_combinations_bin, all_emb, metric='cityblock') 
                density=(out==1).sum(axis=1) # find every niches within a distance of 1
                density=density*(out.min(axis=1)!=0) # remove already explored niches (sampling weight = 0)
                norm= np.sum(density)
                if norm == 0.:
                    norm=1
                density_norm=density/norm

                idx_niches_sampled=np.random.choice(len(skill_combinations_bin),p=density_norm)
                binary_vectors_sampled=skill_combinations_bin[idx_niches_sampled]
                target_skill=list(binary_vectors_sampled)
                target_skill = [int(element) for element in target_skill]
                return target_skill
            case 'none':
                skill_targeted = []
        return skill_targeted
    
    def sample_from_trainset(self,n_examples):
        return list(self.rng.choice(self.env.archive_P3puzzle,size=n_examples))

    def sample_examples(self,trainset_only=False, mode_skill_sampling=None):
        """Sample a batch of examples from the map."""

        n_fewshot = self.config.n_fewshot_examples 
        list_archive_index = []

        skill_targeted = self.skill_sampling(mode_skill_sampling)

        if trainset_only: # only use example from trainset
            list_few_shot_example_phenotypes = list(self.rng.choice(self.env.archive_P3puzzle,size=n_fewshot))
            return (list_few_shot_example_phenotypes, skill_targeted)
        


        if skill_targeted == []: # no target
            #choose random cells 
            liste_niche_idx = self.rd_niche_selection_explored(n=n_fewshot)
            # select n_fewshot examples
            
            for niche_idx in liste_niche_idx:
                archive_indexs = self.sample_examples_from_niche(niche_idx)
                list_archive_index.append(archive_indexs)

        else: # imgep mode
            #all niches explored
            all_emb = list(self.nonzero.keys())
            all_emb = np.array([list(eval(i)) for i in all_emb])

            list_coord_niches_sampled = []
            
            # compute distance between all cells explored and the target cell
            dists = cdist([skill_targeted], all_emb)[0]

            # shuffle indices to have true uniform sampling of closest niches
            # (otherwise, if two niches are at the same distance, the first one will be always sampled)
            shuffled_indices = np.arange(len(dists))
            np.random.shuffle(shuffled_indices)
            nearest_niches = shuffled_indices[np.argsort(dists[shuffled_indices])]
            
            for idx in nearest_niches:
                niche_idx = list(self.nonzero.keys())[idx]
                if not(niche_idx in list_coord_niches_sampled):
                    list_coord_niches_sampled.append(niche_idx)
                    archive_indexs = self.sample_examples_from_niche(niche_idx)
                    list_archive_index.append(archive_indexs)
                if len(list_archive_index)>=n_fewshot:
                    break
        list_few_shot_example_phenotypes = [self.genomes[idx] for idx in list_archive_index]
        return (list_few_shot_example_phenotypes, skill_targeted)

    def search(self, init_steps: int, total_steps: int, atol: float = 0.0) -> str:
        """
        Run the MAP-Elites search algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """

        print(f'use_preprocessed_trainset {self.env.config.use_preprocessed_trainset}')
        print(f'len archive puzzle {len(self.env.archive_P3puzzle)}')
        print(f'load snapshot map {self.config.loading_snapshot_map}')

        # bad coding but huge time gain at initialization should remove it to be compatible with openELM
        if "p3_probsol_Chat" in self.env.config.env_name and not self.config.loading_snapshot_map:
            # add trainset to the MAP
            print("loading P3 trainset to map")
            if self.env.config.use_preprocessed_trainset:
                for individual in self.env.archive_P3puzzle:
                    # self.update_map(individual, 0., 0.)
                    map_ix = self.to_mapindex(individual.emb)
                    # TODO compute quality
                    self.fitnesses.append(self.env.fitness(individual)) # need to add quality here individual.quality
                    self.genomes[map_ix] = individual
                    key_embedding = str(map_ix)
                    value_nonzero = len(self.fitnesses)-1
                    if key_embedding in self.nonzero:
                        self.nonzero[str(map_ix)].append(value_nonzero)
                    else:
                        self.nonzero[str(map_ix)]=[value_nonzero]

        print("number of niched filled from the training set ",self.niches_filled())
        start_step = int(self.start_step)
        total_steps = int(total_steps)
        print(f"start step = {start_step}, total step ={total_steps}, init steps {init_steps}")
        tbar = trange(start_step, total_steps, initial=start_step, total=total_steps)
        if self.niches_filled() == 0:
            max_fitness = -np.inf
            max_genome = None
        else:  # take max fitness in case of filled loaded snapshot
            max_fitness = self.max_fitness()
            max_index = np.argmax(self.fitnesses)
            max_genome = self.genomes[max_index]
        if self.save_history:
            self.history = defaultdict(list)
        
        
        #                           Main loop
        
        for n_steps in tbar:
            print(f'nsteps {n_steps}')
            # print(f'genomes {self.genomes}')
            self.env.idx_generation = n_steps
            self.env.all_phenotypes = self.__getallitems__() # need remove that
            self.env.n_steps = n_steps
            if self.genomes.empty:
                raise ValueError("Empty map after init")
            
            batch: list[Genotype] = []
            if n_steps < init_steps: #or self.genomes.empty:
                print('init steps')
                # Initialise by generating initsteps random solutions.
                # If map is still empty: force to do generation instead of mutation.
                # TODO: use a separate sampler, move batch size to qd config.
                for _ in range(self.env.batch_size):
                    batch.append(self.random_selection(trainset_only=True))
                new_individuals: list[Genotype] =  self.env.random(batch)
            else:
                print('select')
                # Randomly select a batch of elites from the map.
                
                # if self.config.crossover:
                #     crossover_parents = []
                #     previous_ix = None
                #     for i in range(self.config.crossover_parents):
                #         map_ix = self.random_selection()
                #         if map_ix != previous_ix:
                #             crossover_parents.append(self.genomes[map_ix])
                #             previous_ix = map_ix
                #     batch.append(crossover_parents)
                # else:
                for _ in range(self.env.batch_size):
                    batch.append(self.random_selection(trainset_only=False))

                # print(f'BATCH {batch}')
                # Mutate the elite.
                new_individuals = self.env.mutate(batch)

            print('update map')
            max_genome, max_fitness = self.update_map(
                new_individuals, max_genome, max_fitness
            )
            print('done')
            tbar.set_description(f"{max_fitness=:.4f}")
            # print(f'Fitnesses: {self.fitnesses}')

            self.fitness_history["max"].append(self.max_fitness())
            self.fitness_history["min"].append(self.min_fitness())
            self.fitness_history["mean"].append(self.mean_fitness())
            self.fitness_history["qd_score"].append(self.qd_score())
            self.fitness_history["niches_filled"].append(self.niches_filled())
            if (
                self.save_snapshot_interval is not None
                and n_steps != 0
                and n_steps % self.save_snapshot_interval == 0
            ):
                print('save results')
                self.save_results(step=n_steps)
                print('done')

        self.current_max_genome = max_genome
        self.save_results(step=n_steps)
        self.visualize()
        return str(max_genome)

    def update_map(self, new_individuals, max_genome, max_fitness):
        """
        Update the map if new individuals achieve better fitness scores.

        Args:
            new_individuals (list[Genotype]) : List of new solutions
            max_fitness : current maximum fitness

        Returns:
            max_genome : updated maximum genome
            max_fitness : updated maximum fitness

        """
        # `new_individuals` is a list of generation/mutation. We put them
        # into the behavior space one-by-one.
        for individual in new_individuals:

            # print(f"Individual {individual}")

            fitness = self.env.fitness(individual)
            condition_add_individual = fitness != -np.inf
            if condition_add_individual:
                phenotype = individual.to_phenotype()
                map_ix = self.to_mapindex(phenotype)
            else:
                map_ix=None
                phenotype=None
                individual.emb = None
            
            if self.save_all_individual: # save all individuals
                state_individual = individual.__getstate__().copy()
                if "result_obj" in state_individual:
                    del state_individual["result_obj"]
                # if isinstance(state_individual["result_obj"], ExecResult):
                #     state_individual["result_obj"]=str(state_individual["result_obj"])
                if "config" in state_individual:
                    del state_individual["config"]
                if "baseline_emb" in state_individual:
                    del state_individual["baseline_emb"]
                if "emb" in state_individual and isinstance(state_individual["emb"], np.ndarray):
                    state_individual["emb"] = state_individual["emb"].tolist()
                if map_ix!=None:
                    state_individual["map_ix"] = list((int(i) for i in map_ix)) # not sure about that for multi dim map
                else:
                    state_individual["map_ix"] = None
                state_individual["fitness"] = fitness

                self.list_of_all_individuals.append(state_individual)

            if np.isinf(fitness):
                continue
                
            # if the return is None, the individual is invalid and is thrown
            # into the recycle bin.
            if map_ix is None:
                self.recycled[self.recycled_count % len(self.recycled)] = individual
                self.recycled_count += 1
                continue

            if self.save_history:
                # TODO: thresholding
                self.history[map_ix].append(individual)

            

            # If new fitness greater than old fitness in niche, replace.
            try:
                if condition_add_individual: #fitness >= self.fitnesses[map_ix]:
                    # self.fitnesses[map_ix] = fitness
                    # self.genomes[map_ix] = individual
                    key_embedding = str(map_ix)
                    value_nonzero = len(self.fitnesses)
                    if key_embedding in self.nonzero:
                        self.nonzero[str(map_ix)].append(value_nonzero)
                    else:
                        self.nonzero[str(map_ix)]=[value_nonzero]
                    self.fitnesses.append(fitness)
                    self.genomes.archive.append(individual)
            except Exception as e:
                
                print(f"An exception occurred: {str(e)}")
                # print("fitness_map",np.array(self.fitnesses[map_ix]))
                print("map_ix",map_ix)
                # print("fitness_map_shape",np.array(self.fitnesses[map_ix]).shape) #fitness_map_shape (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
                print("fitness",fitness) #fitness 1.0
                print(np.array(map_ix).shape) #(1,)
                print("========================Warning======================")    
                # TODO: pb to fix
                raise ValueError("Error in update map"+str(e))
            # update if new fitness is the highest so far.
            if fitness >= max_fitness: # doesn't work if history >1 can't save multiple individuals in a cell
                max_fitness = fitness
                max_genome = individual

        return max_genome, max_fitness

    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return len(self.nonzero.keys())

    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).max()

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).mean()

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).min()

    def qd_score(self):
        """
        Get the quality-diversity score of the map.

        The quality-diversity score is the sum of the performance of all solutions
        in the map.
        """
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).sum()

    def save_results(self, step: int):
        # create folder for dumping results and metadata
        output_folder = Path(self.config.output_dir) / f"step_{step}"
        os.makedirs(output_folder, exist_ok=True)
        maps = {
            "fitnesses": self.fitnesses,
            "genomes": self.genomes,
            "nonzero": self.nonzero,
        }
        # Save maps as pickle file
        try:
            with open((output_folder / "maps.pkl"), "wb") as f:
                pickle.dump(maps, f)
        except Exception:
            pass
        if self.save_history:
            with open((output_folder / "history.pkl"), "wb") as f:
                pickle.dump(self.history, f)

        with open((output_folder / "fitness_history.pkl"), "wb") as f:
            pickle.dump(self.fitness_history, f)

        # save numpy rng state to load if resuming from deterministic snapshot
        if self.save_np_rng_state:
            rng_generators = {
                "env_rng": self.env.get_rng_state(),
                "qd_rng": self.rng,
            }
            with open((output_folder / "np_rng_state.pkl"), "wb") as f:
                pickle.dump(rng_generators, f)

        # save env_name to check later, for verifying correctness of environment to run with snapshot load
        tmp_config = dict()
        tmp_config["env_name"] = self.env.config.env_name 
        tmp_config["model_name"] = self.config.model_path 
        tmp_config["seed"] = self.env.config.seed
        # tmp_config["eval_k"] = self.env.config.eval_k 
        tmp_config["embedding_model"] = self.env.config.embedding_model_path
        tmp_config["batch_size"] = self.env.config.batch_size
        tmp_config["init_steps"] = self.config.init_steps
        tmp_config["total_steps"] = self.config.total_steps
        if self.env.config.env_name == "p3_probsol_Chat":
            tmp_config["GPT_feedback"] = self.env.config.GPT_feedback
            tmp_config["IMGEP_mode"] = self.env.config.IMGEP_mode
        if self.config.qd_name == "cvtmapelites":
            with open((output_folder / "centroids.npy"), "wb") as f:
                np.save(f, self.centroids)

                pickle.dump(self.centroids, f)
        with open((output_folder / "config.json"), "w") as f:
            json.dump(tmp_config, f,indent=4)
            
        if self.save_all_individual:
            try :
                with open((output_folder / "save_all.json"), "w") as f:
                    json.dump(self.list_of_all_individuals, f,indent=4)
            except Exception:
                pass          
                  
    def plot_fitness(self):
        
        import matplotlib.pyplot as plt
        try: 
            save_path: str = self.config.output_dir
            plt.figure()
            plt.plot(self.fitness_history["max"], label="Max fitness")
            plt.plot(self.fitness_history["mean"], label="Mean fitness")
            plt.plot(self.fitness_history["min"], label="Min fitness")
            plt.legend()
            plt.savefig(f"{save_path}/MAPElites_fitness_history.png")
            plt.close("all")

            plt.figure()
            plt.plot(self.fitness_history["qd_score"], label="QD score")
            plt.legend()
            plt.savefig(f"{save_path}/MAPElites_qd_score.png")
            plt.close("all")

            plt.figure()
            plt.plot(self.fitness_history["niches_filled"], label="Niches filled")
            plt.legend()
            plt.savefig(f"{save_path}/MAPElites_niches_filled.png")
            plt.close("all")

            if len(self.map_dims) > 1:
                if len(self.fitnesses.dims) == 2:
                    map2d = self.fitnesses.latest
                    print(
                        "plotted genes:",
                        *[str(g) for g in self.genomes.latest.flatten().tolist()],
                    )
                else:
                    ix = tuple(np.zeros(max(1, len(self.fitnesses.dims) - 2), int))
                    map2d = self.fitnesses.latest[ix]

                    print(
                        "plotted genes:",
                        *[str(g) for g in self.genomes.latest[ix].flatten().tolist()],
                    )

                plt.figure()
                plt.pcolor(map2d, cmap="inferno")
                plt.savefig(f"{save_path}/MAPElites_vis.png")
            plt.close("all")
        except Exception as e:
            print(e)
            

    def visualize_individuals(self):
        """Visualize the genes of the best performing solution."""
        import matplotlib.pyplot as plt
        try:
            tmp = self.genomes.array.reshape(self.genomes.shape[0], -1)

            # if we're tracking history, rows will be the history dimension
            # otherwise, just the first dimension of the map
            plt.figure()
            _, axs = plt.subplots(nrows=tmp.shape[0], ncols=tmp.shape[1])
            for genome, ax in zip(tmp.flatten(), axs.flatten()):
                # keep the border but remove the ticks
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                try:
                    genome.visualize(ax=ax)
                except AttributeError:
                    pass
            save_path: str = self.config.output_dir
            plt.savefig(f"{save_path}/MAPElites_individuals.png")
        except Exception as e:
            print(e)

class MAPElites(MAPElitesBase):
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: MAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (MAPElitesConfig): The configuration for the algorithm.
        """
        self.map_grid_size = config.map_grid_size
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Set up the discrete behaviour space for the algorithm."""
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*self.env.behavior_space, self.map_grid_size[0] + 1)[1:-1].T  # type: ignore

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return self.map_grid_size * self.env.behavior_ndim

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()




class CVTMAPElites(MAPElitesBase):
    """
    Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

    This replaces the grid of niches in MAP-Elites with niches generated using a
    Centroidal Voronoi Tessellation. Unlike in MAP-Elites, we have a fixed number
    of total niches rather than a fixed number of subdivisions per dimension.
    """

    def __init__(
        self,
        env,
        config: CVTMAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (CVTMAPElitesConfig): The configuration for the algorithm.
        """
        self.cvt_samples: int = config.cvt_samples
        self.n_niches: int = config.n_niches
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self,scale=0.12):
        """Discretize behaviour space using CVT."""
        # lower and upper bounds for each dimension
        if self.config.load_centroids:
        # load centroids in self.centroids from pickle files
            log_path = Path(self.config.log_snapshot_dir)
                
            with open((log_path / "centroids.npy"), "rb") as f:
                self.centroids = np.load(f)
        else:            
            low = self.env.behavior_space[0]
            high = self.env.behavior_space[1]
            state = self.env.__dict__.copy()
            if "archive_P3puzzle" in state:
                list_emb=np.array([p.emb for p in state["archive_P3puzzle"]])
                n_vect2add= self.cvt_samples #- len(list_emb)
                liste_choice=self.rng.choice(len(list_emb), n_vect2add,replace= True)
                embed2nois = list_emb[liste_choice]
                
                noise = self.rng.normal(loc=0.0, scale=scale,size=(embed2nois.shape[0],embed2nois.shape[1]))
                noisy_embed = embed2nois + noise  # add noise to embeddings
                noisy_embed = noisy_embed / np.linalg.norm(noisy_embed, axis=1)[:, np.newaxis] # normalize embeddings
                points = noisy_embed
                # points = np.vstack((list_emb,noisy_embed))
            else:
                points = np.zeros((self.cvt_samples, self.env.behavior_ndim))
                for i in range(self.env.behavior_ndim):
                    points[:, i] = self.rng.normal(low[i], high[i], size=self.cvt_samples)

            k_means = KMeans(init="k-means++", n_init="auto", n_clusters=self.n_niches,random_state=self.config.seed)
            k_means.fit(points)
            self.centroids = k_means.cluster_centers_

        self.plot_centroids(points, k_means)


    def __getallitems__(self) -> list[Phenotype]:
        """
        Returns all the phenotypes that are in the CVT Map."""

        return self.genomes.archive

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return (self.n_niches,)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - self.centroids, axis=1)),)
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()
        self.plot_behaviour_space()

    def plot_centroids(self, points, k_means):
        """
        Plot the CVT centroids and the points used to generate them.

        Args:
            points (np.ndarray, int): the points used to generate the centroids
            k_means (sklearn.cluster.KMeans): the k-means object used to generate the centroids
        """
        import matplotlib.pyplot as plt

        plt.figure()
        labels = k_means.labels_
        if self.env.behavior_ndim == 2:
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )
                plt.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    s=10,
                    marker=".",
                    color=color,
                )
        elif self.env.behavior_ndim >= 3:
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )
                ax.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    points[labels == i, 2],
                    s=10,
                    marker=".",
                    c=[color],
                )
        else:
            print("Not enough dimensions to plot centroids")
            return
        save_path: str = self.config.output_dir
        plt.savefig(f"{save_path}/MAPElites_centroids.png")

    def plot_behaviour_space(self):
        """Plot the first two dimensions (or three if available) of the behaviour space, along with the CVT centroids."""
        import matplotlib.pyplot as plt

        if self.env.behavior_ndim == 2:
            plt.figure()
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )

                # get the first two dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:2]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        plt.scatter(
                            hist[:, 0], hist[:, 1], s=10, marker=".", color=color
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        plt.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            s=10,
                            marker=".",
                            color=color,
                        )

            plt.xlim([0, self.env.behavior_space[1, 0]])
            plt.ylim([0, self.env.behavior_space[1, 1]])

        elif self.env.behavior_ndim >= 3:
            plt.figure()
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )

                # get the first three dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:3]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        ax.scatter(
                            hist[:, 0],
                            hist[:, 1],
                            hist[:, 2],
                            s=10,
                            marker=".",
                            c=[color],
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        ax.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            g.to_phenotype()[2],
                            s=10,
                            marker=".",
                            c=[color],
                        )

            ax.set_xlim([0, self.env.behavior_space[1, 0]])
            ax.set_ylim([0, self.env.behavior_space[1, 1]])
            ax.set_zlim([0, self.env.behavior_space[1, 2]])

        else:
            print("Not enough dimensions to plot behaviour space history")
            return
        save_path: str = self.config.output_dir
        plt.savefig(f"{save_path}/MAPElites_behaviour_history.png")

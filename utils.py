import os
import json 
import numpy as np
import pickle
from tqdm import tqdm
import torch

def getallitems(maps):
    """
    Returns all the phenotypes that are in the Map."""
    genomes = maps["genomes"]
    valid_phenotype=[]
    for gen in np.ndindex(genomes.shape):
        value_gen = type(genomes[gen])
        if value_gen!=float and value_gen!=int:
            valid_phenotype.append(genomes[gen])

    return valid_phenotype

# init maps
def return_cells_filled_per_gen_map_elite(path_save_all,max_gen=-1,include_trainset=False,include_full_trainset=False):
    #_init_discretization():
    n_skills=10
    behavior_space= np.repeat([[0, 1]], n_skills, axis=0).T

    bins = np.linspace(*behavior_space,  3)[1:-1].T  # type: ignore
    # bins
    def to_mapindex(b, bins=bins) :
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )
        

    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems=getallitems(maps)
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]
    if "/elm/" in path_save_all and "run_saved" in path_save_all:
        elm_phenotype = path_save_all.split("maps.pkl")[0]+"_phenotype.npy"
        phen= np.load(elm_phenotype)
        for i in range(len(items_gen)):
            items_gen[i].emb=phen[i]
    print(len(items_gen))
    
    nonzero=np.zeros(shape=[2]*n_skills,dtype=bool)#np.zeros_like(maps["nonzero"]) 
    # list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
            for map_ix_train in list_map_ix_train:
                nonzero[map_ix_train] = True
    # elif include_trainset:
    #     for map_ix_train in list_map_ix_train:
    #         nonzero[map_ix_train] = True

    # separate items per generation
    list_gens= [puzz.idx_generation for puzz in items_gen]
    if max_gen==-1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen<len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix=to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
        number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled


def return_cells_filled_per_puz_gen_map_elite(path_save_all,max_gen=-1,include_trainset=False,include_full_trainset=False):
    #_init_discretization():
    n_skills=10
    behavior_space= np.repeat([[0, 1]], n_skills, axis=0).T

    bins = np.linspace(*behavior_space,  3)[1:-1].T  # type: ignore
    # bins
    def to_mapindex(b, bins=bins) :
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )
        

    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems=getallitems(maps)
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]
    if "/elm/" in path_save_all and "run_saved" in path_save_all:
        elm_phenotype = path_save_all.split("maps.pkl")[0]+"_phenotype.npy"
        phen= np.load(elm_phenotype)
        for i in range(len(items_gen)):
            items_gen[i].emb=phen[i]
    print(len(items_gen))
    
    nonzero=np.zeros(shape=[2]*n_skills,dtype=bool)#np.zeros_like(maps["nonzero"]) 
    # list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
            for map_ix_train in list_map_ix_train:
                nonzero[map_ix_train] = True
    # elif include_trainset:
    #     for map_ix_train in list_map_ix_train:
    #         nonzero[map_ix_train] = True

    # separate items per generation
    list_gens= [puzz.idx_generation for puzz in items_gen]
    if max_gen==-1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen<len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix=to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
            number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled



def return_cells_filled_in_embspace_from_NLPembspace(path_save_all,path_centroids,centroids=None,max_gen=-1,include_trainset=False,include_full_trainset=False,model=None,tokenizer=None,pipeline=None):
    #_init_discretization():
    if centroids is None:
        centroids=np.load(path_centroids)
    # bins
    def to_mapindex(b,centroids=centroids):
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - centroids, axis=1)),)
        )
        
    path_maps_cvt = path_centroids.split("centroids.npy")[0]+"maps.pkl"
    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps_cvt, "rb") as f:
        maps_cvt = pickle.load(f)
    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems=getallitems(maps)
    with torch.no_grad():
        for i in tqdm(range(len(allitems))):
            program_str=allitems[i].program_str
            if pipeline is None:
                inputs = tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                emb = model(inputs.to("cuda"))[0]
                allitems[i].emb=emb.to("cpu").numpy()
            else:
                features = np.array(pipeline(program_str))
                allitems[i].emb=features.mean(axis=1).flatten()
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]
    

            
    print(len(items_gen))
    


    nonzero=np.zeros_like(maps_cvt["nonzero"]) 
    list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            # list_puzzle_full_trainset["program_str"]
        with torch.no_grad():
            for i in tqdm(range(len(list_puzzle_full_trainset))):
                program_str=list_puzzle_full_trainset[i]["program_str"]
                if pipeline is None:
                    inputs = tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                    emb = model(inputs.to("cuda"))[0]
                    emb=emb.to("cpu").numpy()
                else:
                    features = np.array(pipeline(program_str))
                    emb=features.mean(axis=1).flatten()
                list_puzzle_full_trainset[i]["emb"]=emb
        list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
        for map_ix_train in list_map_ix_train:
            nonzero[map_ix_train] = True
              
    elif include_trainset:
        for map_ix_train in list_map_ix_train:
            nonzero[map_ix_train] = True

    # separate items per generation
    list_gens= [puzz.idx_generation for puzz in items_gen]
    if max_gen==-1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen<len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix=to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
        number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled


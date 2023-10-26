import matplotlib.pyplot as plt
import numpy as np
import json

def plot_niche(ax, data, rep_dim=10, name=''):
    assert rep_dim % 2 == 0
    n_gen = len(data)

    # track time of discovery and count in the cell
    discovery = dict()
    for i_gen, gen_data in enumerate(data):
        for datapoint in gen_data:
            assert len(datapoint) == rep_dim
            if tuple(datapoint) not in discovery.keys():
                discovery[tuple(datapoint)] = dict(first_discovery=i_gen+1, count=1)
            else:
                discovery[tuple(datapoint)]['count'] += 1

    # format as image
    dim_img = int(np.sqrt(2 ** rep_dim))
    scales = np.array([2**dim for dim in np.flip(np.arange(rep_dim // 2))])
    img_discovery = np.zeros([dim_img, dim_img])
    img_discovery.fill(np.nan)
    img_count = np.zeros([dim_img, dim_img])
    for datapoint, datapoint_info in discovery.items():
        x = np.dot(np.array(datapoint)[np.arange(0, rep_dim, 2)], scales)
        y = np.dot(np.array(datapoint)[np.arange(1, rep_dim, 2)], scales)
        img_discovery[x, y] = datapoint_info['first_discovery']
        img_count[x, y] = datapoint_info['count']

    im = ax.imshow(img_discovery*30, vmin=0, vmax=1500*30)
    # cbar = plt.colorbar(mappable=im, ax=ax)  # Pass the mappable directly
    plt.axis('off')
    ax.axis('off')
    ax.set_xlim((-0.75, dim_img-0.25))
    ax.set_ylim((-0.75, dim_img-0.28))
    linewidths = np.array(np.flip(np.arange(2, 2*len(scales)+1, 2)))/1.8
    for i_dim in range(rep_dim // 2):
        vals = np.arange(0, dim_img+0.1, scales[i_dim])
        ax.vlines(vals - 0.5, ymin=-0.7, ymax=dim_img+0.5, linewidth=linewidths[i_dim], colors='k')
        ax.hlines(vals - 0.5, xmin=-0.7, xmax=dim_img+0.5, linewidth=linewidths[i_dim], colors='k')
    ax.set_title(name, fontsize=20)
    return im  
    # plt.show()



if __name__ == '__main__':
    #seed 2
    fig, axs = plt.subplots(3, 2, figsize=(8, 14))
    
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  
    path_elm = "/home/flowers/project/OpenELM/run_saved/elm/23-09-18_11_50/step_1499/maps.json"
    path_elm_nlp = "/home/flowers/project/OpenELM/run_saved/elm_nlp/23-09-17/step_1499/maps.json"
    path_im_rd = "/home/flowers/project/OpenELM/run_saved/imgep_random/23-09-16_20_32/step_1500/maps.json"
    # path_im_smart = "/home/flowers/project/OpenELM/run_saved/imgep_smart/23-09-15_21_13/step_1500/maps.json"
    path_rd_gen = "/home/flowers/project/OpenELM/run_saved/random_gen/23-09-18_11_46/step_1499/maps.json"
    path_array=[path_elm, path_elm_nlp, path_im_rd,  path_rd_gen]
    path_name=['ELM', 'ELM semantic', 'ACES',  'Static Gen']
    for i,path in enumerate(path_array):
        with open(path, 'r') as f:
            archive = json.load(f)
    
        # Sort the archive by idx_generation
        sorted_archive = sorted(archive, key=lambda x: x["idx_generation"])

        # Group embeddings by idx_generation
        grouped_embeddings = {}
        for item in sorted_archive:
            if item["idx_generation"] not in grouped_embeddings:
                grouped_embeddings[item["idx_generation"]] = []
            grouped_embeddings[item["idx_generation"]].append(item["emb"])

        # Convert to a list of lists in ascending order of idx_generation
        ax = axs[i // 2, i % 2]  
        grouped_list = [np.array(grouped_embeddings[key],dtype=int) for key in sorted(grouped_embeddings.keys())]


        im = plot_niche(ax, grouped_list, name=path_name[i])

    path_train = "/home/flowers/project/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
    ax = axs[2, 0]  
    with open(path_train, 'r') as f:
        archive = json.load(f)
    grouped_list=[np.array([puzz["emb"] for puzz in archive],dtype=int)]
    plot_niche(ax, grouped_list, name="P3 train")
    path_test= "/home/flowers/project/OpenELM/_phenotype_test.json"
    ax = axs[2, 1]
    with open(path_test, 'r') as f:
        archive = json.load(f)
    grouped_list=[np.array([puzz["emb"] for puzz in archive],dtype=int)]
    plot_niche(ax, grouped_list, name="P3 test")
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Adjust these values to position the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Discovery Generation', rotation=270, labelpad=15)
    fig.subplots_adjust( wspace=0.1, hspace=0.05)
    # ax.tight_layout()
    plt.savefig('./savefig/subplots.png')  # This will save the entire 2x3 grid of plots
    plt.savefig('./savefig/subplots.pdf')  # This will save the entire 2x3 grid of plots

    # plt.show()

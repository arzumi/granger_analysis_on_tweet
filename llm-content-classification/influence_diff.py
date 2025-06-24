import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

def plot_heatmaps_v2(p_value_per_type, k, name):
    type_keys = list(p_value_per_type.keys())
    
    first_type_key = type_keys[0]
    deltas_in_first_type = list(p_value_per_type[first_type_key].keys())
    
    first_delta = deltas_in_first_type[0]
    communities_in_first_delta = list(p_value_per_type[first_type_key][first_delta].keys())
    
    lag_labels = [str(i) for i in range(1, 8)]
    
    for community in communities_in_first_delta:
        for tkey in type_keys:
            M_consumer = np.zeros((7, 7))

            for col, delta_str in enumerate(deltas_in_first_type):
                community_dict = p_value_per_type.get(tkey, {}).get(delta_str, {}).get(community, {})
                
                diff_vals = community_dict.get("diff", [])
                
                for row in range(7):
                    val_consumer = diff_vals[row] if row < len(diff_vals) else np.nan
                    M_consumer[row, col] = val_consumer

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, squeeze=False)
            axes = axes.flatten()  # Now axes is a 1D array with two elements
            axes[1].axis('off')

            # LEFT: 'Core drive Consumer'
            sns.heatmap(
                M_consumer,
                cmap="RdBu_r",
                center=0,
                annot=True,
                fmt=".5f",
                annot_kws={"size": 8},
                ax=axes[0],
                xticklabels=deltas_in_first_type,
                yticklabels=lag_labels,
                cbar=True
            )
            axes[0].set_title(f"{community} - {tkey}\nDifference in influence")
            axes[0].set_xlabel("Window Size (Δ)")
            axes[0].set_ylabel("Lag")


            # add a supertitle
            plt.suptitle(f"P-value Heatmaps (Lag vs. Δ)\nCommunity: {community}, Type: {tkey}, K: {k}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # save figure
            file_name = f"hypothesis/h3/producer_core_values/rank10/{name}_heatmap.pdf".replace(" ", "_")
            plt.savefig(file_name, dpi=300, format='pdf')
            plt.close(fig)

    print("Heatmaps for each (type, community) combination were generated and saved.")

if __name__ == "__main__":
    # for rank in range(1, 4):
        types = {
            "astronomy_expanded": ["top4"], 
            "climate_change_expanded": ["top3"], 
            "comics_expanded": ["top3"], 
            "game_development": ["top3"], 
            "math": ["top_consumer"], 
            "poetry_expanded": ["top4"]
        }
        communities = ["comics_expanded", "astronomy_expanded", "math", "game_development", "climate_change_expanded", "poetry_expanded"]
        # for community in communities:
        #     for k in range(1, 10):
        #         for type in types[community]:
        #             diff = {type: {}}
        #             consumer_core_file = f"hypothesis/h3/consumer_core_values/rank10/{community}_consumer_core_{k}.json"
        #             with open(consumer_core_file, "r") as file:
        #                 consumer_core_data = json.load(file)
                    
        #             for delta in range(1, 8):
        #                 diff[type][delta] = {community: {}}
        #                 consumer_temp = consumer_core_data[type][str(delta)][community]
                        
        #                 diff[type][delta][community]["diff"] = [consumer_temp["Consumer drive Core"][i] - consumer_temp["Core drive Consumer"][i] for i in range(0, 7)]

        #             plot_heatmaps_v2(diff, k, f"diff_{k}_{type}_{community}_consumer_core")
        #             with open(f"{community}_consumer_core_diff_{k}.json", "w") as file:
        #                 json.dump(diff, file)
        
        for community in communities:
            for k in range(1, 10):
                for type in types[community]:
                    diff = {type: {}}
                    consumer_core_file = f"hypothesis/h3/producer_core_values/rank6/{community}_producer_core_{k}.json"
                    with open(consumer_core_file, "r") as file:
                        consumer_core_data = json.load(file)
                    
                    for delta in range(1, 8):
                        diff[type][delta] = {community: {}}
                        consumer_temp = consumer_core_data[type][str(delta)][community]
                        
                        diff[type][delta][community]["diff"] = [consumer_temp["Core drive Producer"][i] - consumer_temp["Producer drive Core"][i] for i in range(0, 7)]

                    plot_heatmaps_v2(diff, k, f"diff_{k}_{type}_{community}")
                    with open(f"hypothesis/h3/producer_core_values/rank6/{community}_producer_core_diff_{k}.json", "w") as file:
                        json.dump(diff, file)
            
                        

            

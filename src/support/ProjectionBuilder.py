import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
import os

class ProjectionBuilder():

    def __init__(self, df: pd.Series, backlog: str):

        self.original_value_counts = df.value_counts()
        self.backlog = backlog.replace(' ', '_')
        self.id_session = f'{self.backlog}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        self.mapping_COD2class ={ 1: 1,
                                    2: 2,
                                    3: 3,
                                    5: 4,
                                    8: 5,
                                    13: 6,
                                    21: 7,
                                    40: 8,
                                    100: 9
                                }

        self.mapping_class2COD ={ 1: 1,
                                    2: 2,
                                    3: 3,
                                    4: 5,
                                    5: 8,
                                    6: 13,
                                    7: 20,
                                    8: 40,
                                    9: 100
                                }
        
        # how many classes I expected to have
        self.classes = list(self.mapping_class2COD.keys())
        # fix unbalanced classes in case a dataset doesn't present any data point with a certain class
        self.adjusted_value_counts = self.fix_unbalanced_classes()
        # retrieve probability to get a certain class, given the classes distribution found in the original dataset
        self.original_width_vector = self.adjusted_value_counts / self.adjusted_value_counts.sum()     
        
        # adjusting the width in order for the intervals to range from 0 up to 1
        self.adjusted_width_vector = self.original_width_vector.copy() # make a copy of the original that will be adjusted
        self.adjusted_width_vector[0]=0 # introduce 0 index with 0 value
        self.adjusted_width_vector.index = self.adjusted_width_vector.index+1 # shit all values 1 position ahead
        self.intervals_vector = self.adjusted_width_vector.sort_index().cumsum() # cumulative sum for full view of projection space




    def plot_projection_space(self,save_img_path:Optional[str]=None, save_img:bool=False, dont_show_img:bool=False) -> None:

        # Define the intervals
        width_series = self.adjusted_width_vector.sort_index().cumsum()
        width_vector = self.intervals_vector.to_numpy()  # Example interval endpoints

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 2))

        # Plot the main line
        ax.plot([width_vector[0], width_vector[-1]], [0, 0], 'b-', linewidth=2)

        ### TMP COMMENTED OUT ###
        # Add points to highlight intervals
        # for i in width_vector:
        #    ax.plot(i, 0, 'ro', markersize=10)
        ### TMP COMMENTED OUT ###

        # Add vertical lines to emphasize intervals
        for i in width_vector:
            ax.axvline(x=i, color='g', linestyle='--', alpha=0.5)

        # Remove y-axis
        ax.yaxis.set_visible(False)

        # Set y-axis limits to center the line
        ax.set_ylim(-0.5, 0.5)

        # Customize the plot
        ax.set_xlabel('Intervals')
        ax.set_title(f'Projection Space representation of Backlog {self.backlog} according to gold standard distribution')

        # Add interval width labels
        for i in range(len(width_vector) - 1):
            width = self.mapping_class2COD[width_series.index.sort_values()[i]]
            ax.annotate(f'{width}', 
                        xy=((width_vector[i] + width_vector[i+1]) / 2, 0.1),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        va='bottom')

        # don't dispaly image..
        if dont_show_img:
            # ..but save it
            if save_img:
                # ensure img path is provided
                if save_img_path is None and os.path.exists(save_img_path):
                    print('Parameter <save_img_path> does not exists or is not provided. Value provided for aforementioned parameter is : \n{save_img_path}')
                else:
                    plt.savefig(os.path.join(save_img_path,f'ProjectionSpaceImg_{self.id_session}.png'), bbox_inches='tight', dpi=300)
            else:
                return None

        # yes, show the image
        else:

            plt.tight_layout()

            # ..but save it
            if save_img:
                # ensure img path is provided
                if save_img_path is None and os.path.exists(save_img_path):
                    print('Parameter <save_img_path> does not exists or is not provided. Value provided for aforementioned parameter is : \n{save_img_path}')
                else:
                    plt.savefig(os.path.join(save_img_path,f'ProjectionSpaceImg_{self.id_session}.png'), bbox_inches='tight', dpi=300)

            else:
                return None
             
            plt.show()


    def fix_unbalanced_classes(self) -> pd.Series:

        adjusted_value_counts = self.original_value_counts.reindex(self.classes, fill_value=1)

        return adjusted_value_counts

# Import matplotlib & seaborn for charting/plotting
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

# For 3D charts
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.cm import cool


base_color_list = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


save_images = False
default_image_save_location = "d:\\temp\\plots"
default_dpi = 100
  
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def uni_category_barchart(df, column_name, limit_bars_count_to=10000, order_by_label=False):
    
    data_for_chart = df[column_name].value_counts(dropna=False)[:limit_bars_count_to]
    if order_by_label:
        data_for_chart = data_for_chart.sort_index()
            
    ax = data_for_chart.plot(kind='bar',
                                    figsize=(14,8),
                                    title="Uni Categorical [" +column_name+"]")
    ax.set_xlabel("Categories in [" + column_name + "]")
    ax.set_ylabel("No. of Records or Rows per Category")
    add_value_labels(ax)
    if save_images:
        plt.savefig(default_image_save_location + "\\Uni_Barchart_" +column_name + ".png")
    
def uni_continuous_boxplot(df, column_name):
    sns.boxplot(y=df[column_name])    
    if save_images:
        plt.savefig(default_image_save_location + "\\Uni Continuous Boxplot-" +column_name+ ".png")

def bi_continuous_continuous_scatterplot(df, column_name1, column_name2):
    sns.scatterplot(data=df, x=column_name1, y=column_name2)   
    if save_images:
        plt.savefig(default_image_save_location + "\\Bi Continuous Continuous Scatterplot-" +column_name1+" " + column_name2 + ".png")    

def bi_continuous_category_boxplot(df, continuous1, category2): 
    sns.boxplot(y=continuous1, x=category2, data=df)


def multi_continuous_continuous_category_scatterplot(df, column_name1, column_name2, column_name3): 
    sns.scatterplot(data=df, x=column_name1, y=column_name2, hue=column_name3)
    if save_images:
        plt.savefig(default_image_save_location + "\\Multi Continuous Continuous Category Scatterplot-" +column_name1+" " + column_name2 + ".png", dpi = default_dpi)    

def multi_continuous_category_category_boxplot(df, continuous1, category2, category3): 
    sns.boxplot(y=continuous1, x=category2, hue=category3, data=df)


def bi_category_category_crosstab_percentage(df, category_column1, category_column2) :
    ct=pd.crosstab(df[category_column1], df[category_column2])
    ct.div(ct.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
    
def bi_category_category_countplot(df, category_column1, category_column2) :
    sns.countplot(x=category_column1, hue=category_column2, data=df)

def bi_continuous_category_violinplot(df, category1, continuous2) :
    sns.violinplot(x=category1, y=continuous2, data=df, scale="count")

def multi_continuous_category_category_violinplot(df, continuous1, category_column2, category_column3):
    sns.violinplot(y=continuous1, x=category_column2, hue=category_column3, data=df, split=True, palette='muted')

def scaleTo01(x):
    return ((x-min(x))/(max(x)-min(x)))

def get_n_colors(n):
    return[ cool(float(i)/n) for i in range(n) ]
    
def multi_continuous_continuous_continuous_category_scatterplot(df, continuous1, continuous2, continuous3, category4):
    
    cat_unique_list = list(df[category4].unique())
    
    idx = [cat_unique_list.index(x) for x in cat_unique_list]
    colors = [base_color_list[i] for i in idx]
    color_dict = dict(zip(cat_unique_list, colors))

    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')
    
    ax.scatter(df[continuous1], df[continuous2], df[continuous3],  c=df[category4].map(color_dict))
    ax.set_xlabel(continuous1)
    ax.set_ylabel(continuous2)
    ax.set_zlabel(continuous3)
    ax.invert_yaxis()

    # Add legend with proxy artists
    column_pathces = [mpatches.Patch(color=colors[x], label=cat_unique_list[x]) for x in idx]
    plt.legend(handles=column_pathces, title=category4)
    

def multi_continuous_continuous_continuous_scatterplot(df, continuous1, continuous2, continuous3, maintain_same_color_palette=False):
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')
    colors_df = df[[continuous1, continuous2, continuous3]]
    colors_df.columns = ['red', 'green', 'blue']
    if maintain_same_color_palette:
        if max(colors_df['red'])-min(colors_df['red']) > max(colors_df['green'])-min(colors_df['green']):
            colors_df = colors_df.rename(columns={'red': 'green', 'green': 'red'})    
            colors_df.head()
        if max(colors_df['red'])-min(colors_df['red']) > max(colors_df['blue'])-min(colors_df['blue']):
            colors_df = colors_df.rename(columns={'red': 'blue', 'blue': 'red'})    
            colors_df.head()
        if max(colors_df['green'])-min(colors_df['green']) > max(colors_df['blue'])-min(colors_df['blue']):
            colors_df = colors_df.rename(columns={'green': 'blue', 'blue': 'green'})    
            colors_df.head()
    colors_df = colors_df[['red','green','blue']].apply(scaleTo01)
    colors_array = colors_df.values    
    ax.scatter(df[continuous1], df[continuous2], df[continuous3],  facecolors=colors_array)
    ax.set_xlabel(continuous1)
    ax.set_ylabel(continuous2)
    ax.set_zlabel(continuous3)
    ax.invert_yaxis()
    
def univariate_charts(data_frame):
    for column_name in data_frame.columns:        
        if pd.api.types.is_numeric_dtype(data_frame[column_name]):
            uni_continuous_boxplot(data_frame, column_name)
        elif pd.api.types.is_datetime64_dtype(data_frame[column_name]):
            print(column_name + ' Date')
        elif pd.api.types.is_categorical_dtype(data_frame[column_name]) or pd.api.types.is_object_dtype(data_frame[column_name]):
            uni_category_barchart(data_frame, column_name)
        else:
            print(column_name + 'Unknown')
            
            

    
    
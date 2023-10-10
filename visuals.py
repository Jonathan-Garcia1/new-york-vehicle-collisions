import pandas as pd
import numpy as np
import os
import math
import env

import matplotlib.pyplot as plt
import seaborn as sns


def injuries_dist(df):

    # Customized plot for distribution of the target variable 'injuries'
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='injuries', data=df, color="#1f77b4")

    # Remove y-axis and grid
    ax.set(yticklabels=[])
    ax.yaxis.set_ticks_position('none')
    sns.despine(left=True, bottom=True)

    # Get the y-axis height
    y_axis_height = ax.get_ylim()[1]  # This retrieves the maximum y-axis value

    # Display count on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()/1000:.1f}K', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                    ha='center', va='baseline', fontsize=16)

    # Add title and labels with customized padding and font size
    plt.title('Distribution of Injuries', fontsize=12)
    plt.xlabel('Injuries', labelpad=20, fontsize=12)
    plt.ylabel('')

    # Show the plot
    plt.show()



def customized_barplot_v4(x, y, data, title, xlabel, hue=None, hue_order=None, hue_palette=None, single_color=None, sort_data=True, x_label_rotation=0, bar_font_size=12, figsize=(10, 6)):
    if sort_data:
        data_to_plot = data.sort_values(y, ascending=False)
    else:
        data_to_plot = data
    
    plt.figure(figsize=figsize)
    
    if single_color:
        ax = sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order, data=data_to_plot, color=single_color)
    else:
        ax = sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order, palette=hue_palette, data=data_to_plot)
    
    ax.set(yticklabels=[])
    ax.yaxis.set_ticks_position('none')
    sns.despine(left=True, bottom=True)
    
    y_axis_height = ax.get_ylim()[1]
    
    for p in ax.patches:
        if p.get_height() < 1:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                        ha='center', va='baseline', fontsize=bar_font_size)
        elif p.get_height() < 1000:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                        ha='center', va='baseline', fontsize=bar_font_size)
        else:
            ax.annotate(f'{p.get_height()/1000:.1f}K', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                        ha='center', va='baseline', fontsize=bar_font_size)
    
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, labelpad=20, fontsize=12)
    plt.ylabel('')
    plt.xticks(rotation=x_label_rotation)
    
    plt.show()

def customized_barplot_subplot(x, y, data, title, xlabel, ax, x_label_rotation=0, bar_font_size=12):
    sns.barplot(x=x, y=y, data=data.sort_values(y, ascending=False), color="#1f77b4", ax=ax)
    
    ax.set(yticklabels=[])
    ax.yaxis.set_ticks_position('none')
    sns.despine(left=True, bottom=True, ax=ax)
    
    y_axis_height = ax.get_ylim()[1]
    
    for p in ax.patches:
        if p.get_height() < 1000:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                        ha='center', va='baseline', fontsize=bar_font_size)
        else:
            ax.annotate(f'{p.get_height()/1000:.1f}K', (p.get_x() + p.get_width() / 2., p.get_height() + (y_axis_height * 0.01)),
                        ha='center', va='baseline', fontsize=bar_font_size)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, labelpad=20, fontsize=12)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=x_label_rotation)



def customized_horizontal_barplot(x, y, data, title, ylabel, y_label_rotation=0, bar_font_size=12, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, data=data.sort_values(x, ascending=False), color="#1f77b4")
    
    ax.set(xticklabels=[])
    ax.xaxis.set_ticks_position('none')
    sns.despine(left=True, bottom=True)
    
    x_axis_width = ax.get_xlim()[1]
    
    for p in ax.patches:
        if p.get_width() < 1:
            ax.annotate(f'{p.get_width():.2f}', (p.get_width() + (x_axis_width * 0.01), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=bar_font_size)
        elif p.get_width() < 1000:
            ax.annotate(f'{p.get_width():.0f}', (p.get_width() + (x_axis_width * 0.01), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=bar_font_size)
        else:
            ax.annotate(f'{p.get_width()/1000:.1f}K', (p.get_width() + (x_axis_width * 0.01), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=bar_font_size)
    
    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, labelpad=20, fontsize=12)
    plt.xlabel('')
    plt.yticks(rotation=y_label_rotation)
    
    plt.show()

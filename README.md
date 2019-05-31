<details><summary>Chart + Util = Chartil (Click to expand)</summary>


# Chart + Util = Chartil
Data visualization: Simple, Single unified API for plotting and charting

During EDA/data preparation we use few common and fixed set of chart types to analyse the relation among various features. 
Few are simple charts like univariate and some are complex 3D or even multiple features>3.

This api is simple, single api to plot various type of relations which will hide all the technical/code details from Data Science task and approch.
This overcomes the difficulties of maintaining several api or libraries and avoid repeated codes. 

Using this approach we just need one api (Rest all decided by library)

	from KUtils.eda import chartil

    chartil.plot(dataframe, [list of columns]) or
    chartil.plot(dataframe, [list of columns], {optional_settings})


Demo code:

# Load UCI Dataset. Download [From here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease/)
	heart_disease_df = pd.read_csv('../input/uci/heart.csv')


# Quick data preparation
	column_to_convert_to_categorical = ['target', 'cp', 'fbs', 'exang', 'restecg', 'slope', 'ca', 'thal']
	for col in column_to_convert_to_categorical:
		heart_disease_df[col] = heart_disease_df[col].astype('category')
    
	heart_disease_df['age_bin'] = pd.cut(heart_disease_df['age'], [0, 32, 40, 50, 60, 70, 100], labels=['<32', '33-40','41-50','51-60','61-70', '71+'])   

	heart_disease_df['sex'] = heart_disease_df['sex'].map({1:'Male', 0:'Female'})

	heart_disease_df.info()

# Heatmap
	chartil.plot(heart_disease_df, heart_disease_df.columns) # Send all column names 
![Heatmap Numerical](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/heatmap1.png)

	chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'include_categorical':True} ) 
![Heatmap With categorical](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/heatmap2.png)

	chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'include_categorical':True, 'sort_by_column':'trestbps'} ) 
![Heatmap With categorical and ordered by a column](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/heatmap3.png)

	# Force to plot heatmap when you have fewer columns, otherwise tool will decide as different chart
	chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps'], chart_type='heatmap') 
![forced_heatmap](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/forced_heatmap.png)

# Uni-categorical          
	chartil.plot(heart_disease_df, ['target']) # Barchart as count plot 
![Uni Categorical](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/uni_categorical.png)

# Uni-Continuous
	chartil.plot(heart_disease_df, ['age'])
![Uni boxplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/uni_boxplot.png)

	chartil.plot(heart_disease_df, ['age'], chart_type='barchart') # Force barchart on cntinuous by auto creating 10 equal bins 
![Uni barchart_forced](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/uni_barchart_forced.png)

	chartil.plot(heart_disease_df, ['age'], chart_type='barchart', optional_settings={'no_of_bins':5}) # Create custom number of bins 
![Uni uni_barchart_forced_custom_bin_size](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/uni_barchart_forced_custom_bin_size.png)

	chartil.plot(heart_disease_df, ['age'], chart_type='distplot') 
![Uni distplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/uni_distplot.png)

# Uni-categorical with optional_settings
	chartil.plot(heart_disease_df, ['age_bin']) # Barchart as count plot
![Uni distplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/age-bin.png)

	chartil.plot(heart_disease_df, ['age_bin'], optional_settings={'sort_by_value':True})
![Uni distplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/age-bin_sort.png)

	chartil.plot(heart_disease_df, ['age_bin'], optional_settings={'sort_by_value':True, 'limit_bars_count_to':5})
![Uni distplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/age-bin_sort_limit.png)

# Bi Category vs Category (& Univariate Segmented)
	chartil.plot(heart_disease_df, ['sex', 'target'])
![Bi Category](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_category_bar.png)

	chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='crosstab')
![Bi Category](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_category_cross_tab.png)

	chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='stacked_barchart')
![Bi Category](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_category_stackedbar.png)

# Bi Continuous vs Continuous
	chartil.plot(heart_disease_df, ['chol', 'thalach']) # Scatter plot
![Bi Continuous scatter](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_continuous_scatter.png)

# Bi Continuous vs Category
	chartil.plot(heart_disease_df, ['thalach', 'sex']) # Grouped box plot (Segmented univariate)
![Bi continuous_catergory_box](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_continuous_catergory_box.png)

	chartil.plot(heart_disease_df, ['thalach', 'sex'], chart_type='distplot') # Distplot
![Bi continuous_catergory_distplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/bi_continuous_catergory_distplot.png)

# Multi 3 Continuous
	chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps']) # Colored 3D scatter plot
![3 Continuous 3D](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/3continuous_3d.png)

# Multi 3 Categorical
	chartil.plot(heart_disease_df, ['sex', 'age_bin', 'target']) # Paired barchart
![3 paired_3d_grouped_barchart](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/paired_3d_grouped_barchart.png)

# Multi 2 Continuous, 1 Category
	chartil.plot(heart_disease_df, ['chol', 'thalach', 'target']) # Scatter plot with colored groups 
![Grouped Scatter plot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/group_scatter_plot.png)

# Multi 1 Continuous, 2 Category
	chartil.plot(heart_disease_df, ['thalach', 'sex', 'target']) # Grouped boxplot
![Grouped 1continuous_2category_boxplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/1continuous_2category_boxplot.png)

	chartil.plot(heart_disease_df, ['thalach', 'sex', 'target'], chart_type='violinplot') # Grouped violin plot
![Grouped 1continuous_2category_violinplot](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/1continuous_2category_violinplot.png)

# Multi 3 Continuous, 1 category
	chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps', 'target']) # Group Color highlighted 3D plot
![Grouped 3d_scatter](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/grouped_3d_scatter.png)

# Multi 3 category, 2 Continuous
	chartil.plot(heart_disease_df, ['sex','cp','target','thalach','trestbps']) # Paired scatter plot
![Grouped Paired_3d_grouped_scatter](https://raw.githubusercontent.com/KeshavShetty/ds/master/Roughbook/misc_resources/paired_3d_grouped_scatter.png)

</details>


<details><summary>Auto Linear Regression (Click to expand)</summary>
	Todo:
</details>
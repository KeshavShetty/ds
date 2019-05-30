# Chart + Util = Chartil

During EDA/data preparation stage, I use few fixed chart types to analyse the relation among various features. 
Few are simple chart like univariate and some are complex 3D or even multiple features>3.

Over the period it became complex to maintain all relevant codes or repeat codes. 
Instead I developed a simple, single api to plot various type of relations which will hide all technical/code details from Data Science task and approch.

Using this approach I just need one api

from KUtils.eda import chartil

    chartil.plot(dataframe, [list of columns]) or
    chartil.plot(dataframe, [list of columns], {optional_settings})


e.g:

# Heatmap
chartil.plot(uci_heart_disease_df, uci_heart_disease_df.columns) # Send all column names
chartil.plot(uci_heart_disease_df, uci_heart_disease_df.columns, optional_settings={'include_categorical':True} )
chartil.plot(uci_heart_disease_df, uci_heart_disease_df.columns, optional_settings={'include_categorical':True, 'sort_by_column':'trestbps'} )

# Uni-categorical          
chartil.plot(uci_heart_disease_df, ['target']) # Barchart as count plot

# Uni-Continuous
chartil.plot(heart_disease_df, ['age']) # boxplot
chartil.plot(heart_disease_df, ['age'], chart_type='barchart') # Force barchart on cntinuous by auto creating 10 equal bins
chartil.plot(heart_disease_df, ['age'], chart_type='barchart', optional_settings={'no_of_bins':5}) # Create custom number of bins
chartil.plot(heart_disease_df, ['age'], chart_type='distplot')

# Uni-categorical with optional_settings
chartil.plot(heart_disease_df, ['age_bin']) # Barchart as count plot
chartil.plot(heart_disease_df, ['age_bin'], optional_settings={'sort_by_value':True})
chartil.plot(heart_disease_df, ['age_bin'], optional_settings={'sort_by_value':True, 'limit_bars_count_to':5})

# Bi Category vs Category (& Univariate Segmented)
chartil.plot(heart_disease_df, ['sex', 'target'])
chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='crosstab')
chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='stacked_barchart')

# Bi Continuous vs Continuous
chartil.plot(heart_disease_df, ['chol', 'thalach']) # Scatter plot

# Bi Continuous vs Category
chartil.plot(heart_disease_df, ['thalach', 'sex']) # Grouped box plot (Segmented univariate)
chartil.plot(heart_disease_df, ['thalach', 'sex'], chart_type='distplot') # Distplot

# Multi 3 Continuous
chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps']) # Colored 3D scatter plot

# Multi 3 Categorical
chartil.plot(heart_disease_df, ['age_bin', 'sex', 'target']) # Paired barchart

# Multi 2 Continuous, 1 Category
chartil.plot(heart_disease_df, ['chol', 'thalach', 'target']) # Scatter plot with colored groups

# Multi 1 Continuous, 2 Category
chartil.plot(heart_disease_df, ['thalach', 'sex', 'target']) # Grouped boxplot
chartil.plot(heart_disease_df, ['thalach', 'sex', 'target'], chart_type='violinplot') # Grouped violin plot

# Multi 3 Continuous, 1 category
chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps', 'target']) # Group Color highlighted 3D plot

# Multi 3 Continuous, 2 category
chartil.plot(heart_disease_df, ['sex','cp','target','thalach','trestbps']) # Paired wscatterplot


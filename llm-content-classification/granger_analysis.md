# Intro
- In this README, we will go over how to run the granger analysis. 
- You need the following files:
    - openai_granger_analysis_1.py
    - delayed_consumer_core_analysis.py
    - delayed_producer_core_analysis.py
    - influence_diff.py

# First step - get consumer & core topics
- Before running the actual program you need to first store the consumer and core topics in the folder of each community. To do so, open *openai_granger_analysis_1.py* and run the program. Two files will be generated where one is for consumer topics and one for core topics. 

# Second step - granger analysis 
- Each community gives us the best result under different topic type. Hence in the main block, we have the variable *type* to generate different time series for each community based on the topic they produced the best results. 
    - The best result is obtained via running various analysis across topcis and checking which one has the best result. 
- k is the lag, where we are shifting the time series k steps down, delta is the window size, and inside of the granger analysis we run the analysis on 7 lags. 
- There are different locations in the code where you can uncomment to run different experiments, each has a comment on top. 
- Currently, the image that is generated is a histogram but you can uncomment various location of the *plot_heatmap_v2* function to generate different images. 
- All the time series combination and the actual program is in the main block. 
- Each community has a different time range based on the most active periods of the users in the community. However, for all the new communities, they have the same date range as we are looking at a year span. You can find the date range in *community_date_ranges* variable.
- Generated images will be saved to the root folder direclty. Alternatively, you can also save it to respective community folders or any folder you want. 
- The most up to date file is the *delayed_consumer_core_analysis.py* file, the *delayed_producer_core_analysis.py* file is not up to date. Hence, you can modify the producer & core file to make sure it is structured in the same way as the consumer & core file. They are mostly the same, but there are just some differences. 

# Third step - generate difference graph
- To generate the difference graph, you will need to store the values of the experiement. The corresponding code that needs to be uncommented is noted in the file. 
- Use the *influence_diff.py* to generate the heatmap for differences between the two direction of the influence. 
- The two for loops corresponds to the two different analysis, one for consuerm and core, and one for producer and core. 

# By the end of this you should have heatmaps for the p values, and another set of heatmaps for the differences. 
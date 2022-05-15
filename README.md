# Discovery 2

Implementation and comparison of Capsule Networks and Convolutional Neural Networks for classification of Exoplanet Candidates<br>
<br>
![Cover picture](https://github.com/Jon-Flan/Discovery2/blob/main/pictures/githubcover.PNG)

# Introduction 
This project examines the performance of Convolutional Neural Networks and Capsule Networks in the problem domain of classifying exoplanet candidates using light fluctuation readings from NASA telescope data. NASA telescopes record light intensity readings from observed stars, if a planet is orbiting one of these stars on the same visual plane as the telescope, an identifiable dip in light intensity is created. This type of reading is known as a transit event and can take different forms depending on the number and size of planets orbiting a particular star. This method of detection makes up 76.82% of the 5,030 exoplanets discovered.

# Data Retrieval

Data in the form of FITS files containing FLUX readings were directly downloading for 6610 stars using the LightKurve python library in the Get_Data.py script. Each instance had an average collection size of 50,000 over the mission period resulting in approximately 300 million collections of flux readings downloaded locally to the FITS folder created in the project directory (31.3GB). <br>

Each collection was then stitched together using a second script (Pre_Process1.py) to create one continuous flux file for each star. The results were centred and binned into 2000 flux readings for each star. Where data was missing due to the binning process a linear interpolation was applied. The results where then transposed to row instead of column where each row contained one observation of a star, its ID, name, disposition, p disposition and 2000 flux readings. These rows where then concatenated together, normalised, and exported to a single csv file (candidates_with_flux.csv). The disposition and p disposition where then used to create the labelled dataset containing either 1 for confirmed planet or 0 for false positive and 2000 flux readings (labeled_data.csv).  

![Data download](https://github.com/Jon-Flan/Discovery2/blob/main/pictures/doc_pics/flux_down.jpg)

# Data Transformation

Transforming the flux and candidate data as part of the the project involved correctly labelling the data in the context of the machine learning models being applied. The data mining implementation involves binary classification therefore the koi_disposition column which was used to label the flux readings is converted from “confirmed” to 1 and “false positive” to 0.
For the flux readings themselves there are 2,000 readings per star. Resulting in untidy fluctuations that could impact any models used.



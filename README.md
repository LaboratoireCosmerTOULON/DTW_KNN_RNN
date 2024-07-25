# DTW_KNN_RNN

This  github repository is associated with the paper :"Subject-independent diver gesture classification using upper limb movement" submitted to the journal Robotics and Automation Letter and currently under review.

<p align="left">
  <img src="./figures/setup_mini.jpg?raw=false" alt="setup image", title="data acquisition setup", style="width:50%;"/>
</p>
This repository is organized as follow: 

+ ./data/ directory that contains the data used in the paper. 
+ ./DTW-KNN/ include for the DTW classification
+ ./RNN/ includes the RNN-based classification

## DTW-KNN
The subdirectory DTW-KNN contains all the code needed to perform the SVM seperation and the DTW-KNN classification. 
The subdirectory includes the  following folders: 
- yaml/ 
- scripts/
- utils/

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
    <!-- First Row: 3 GIFs -->
    <figure style="flex: 1 1 30%; margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/airshortage.gif" alt="air shortage" style="width: 100%; height: auto;">
        <figcaption>Air shortage</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/assemble.gif" alt="assemble" style="width: 100%; height: auto;">
        <figcaption>Assemble</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/cold.gif" alt="cold gesutre" style="width: 100%; height: auto;">
        <figcaption>Cold</figcaption>
    </figure>
    <!-- Second Row: 3 GIFs -->
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/godown.gif" alt="GIF 4" style="width: 100%; height: auto;">
        <figcaption>Go down</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/goup.gif" alt="GIF 5" style="width: 100%; height: auto;">
        <figcaption>Go up</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/half-pressure.gif" alt="half-pressure" style="width: 100%; height: auto;">
        <figcaption>Half-pressure</figcaption>
    </figure>
    <!-- Third Row: 3 GIFs -->
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/notwell.gif" alt="not well" style="width: 100%; height: auto;">
        <figcaption>Not well</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/ok.gif" alt="GIF 8" style="width: 100%; height: auto;">
        <figcaption>Ok</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/panting.gif" alt="GIF 9" style="width: 100%; height: auto;">
        <figcaption>Panting</figcaption>
    </figure>
    <!-- Fourth Row: 2 GIFs -->
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/reserve.gif" alt="GIF 10" style="width: 100%; height: auto;">
        <figcaption>Reserve</figcaption>
    </figure>
    <figure style="flex: 1 1 30%;  margin:0; box-sizing: border-box; text-align: center;">
        <img src="./figures/stabilize.gif" alt="GIF 10" style="width: 100%; height: auto;">
        <figcaption>Stabilize</figcaption>
    </figure>
</div>

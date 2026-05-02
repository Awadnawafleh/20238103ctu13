Botnet Detection on the CTU-13 Dataset using Unsupervised Learning
AI in Cyber Security – Assignment 1
Princess Sumaya University for Technology
1. Introduction
A botnet consists of infected hosts under the control of an attacker. They are used to perform malicious activities such as spamming, data exfiltration, and DDoS attacks. Detecting botnet-generated traffic can be difficult since attackers constantly evolve their tools. Therefore, signature-based detection will not be sufficient by itself. For this assignment, we will attempt to identify botnet flows from a real network capture using unsupervised machine learning. That is, we will not use the labels during training. Botnet traffic has different behavior than normal traffic and should form its own clusters or be detected as outliers.
2. Dataset
Our data came from CTU-13 dataset (taken from 2011/08/15). CTU-13 dataset is widely known dataset for botnet detection research. This dataset was captured in CTU University in Czech Republic. There are 129,832 flows and 15 attributes in the file we worked with (capture20110815.binetflow). Attributes include duration, protocol, src/dst address and ports, packet count and byte count, etc. Botnet class is highly imbalanced with only 901 Botnet flows (~0.7%) and ~129k Normal/Background flows (~99.3%). The imbalance represents real networks and adds difficulty to our project.
3. Methodology
We implemented three unsupervised learning algorithms: K-Means, DBSCAN, and Isolation Forest. Our pipeline consisted of four steps: 
1) Preprocessing: We selected numeric and basic categorical features (Dur, Proto, Sport, Dport, State, TotPkts, TotBytes, SrcBytes), label encoded Proto and State columns, imputed missing values with 0, and generated two basic ratios BytesPerPkt and PktsPerSec. All columns were standardized with StandardScaler. 
2) Selecting k for K-means: We tested k = [2,6] and determined best cluster number using Elbow method (inertia) and Silhouette score. Based on silhouette score calculated on 10,000 sample points, we selected k = 6 because score was maximized at k = 5 and 6 (0.430).
3) Training: K-means and Isolation Forest were trained on the entire dataset. DBSCAN took too long to run on 130k points so we fit it to a random subset of 30,000 flows. We selected eps = 0.5 and min_samples = 10. 
4) Determining clusters that correspond to botnet: For K-means, we select the cluster containing the largest fraction of known botnet traffic as our predicted botnet cluster. For DBSCAN, we select the -1 noise points as botnet traffic. For Isolation Forest, we mark anomalies as botnet.

4. Results
Table 1 shows the performance of the three algorithms on the labels (the labels were only used for evaluation, not for training)
Table 1. Comparison of the three algorithms
Model	Accuracy	Precision	Recall	F1
K-Means (k=6)	0.761	0.022	0.778	0.043
DBSCAN (sample)	0.942	0.008	0.064	0.015
Isolation Forest	0.984	0.052	0.075	0.062
K-Means achieved the highest Recall (77.8%), indicating it detected most botnet flows. However, its Precision is extremely low (2.2%) because it marks many normal flows as botnet as well. This is expected when data is heavily imbalanced – any big cluster will have many false positives if we simply label the entire cluster as malicious.
DBSCAN eps=0.5 was only able to detect ~6% of botnet flows. DBSCAN performs well when malicious points form small isolated regions that are easily distinguishable from the main density. However, in this dataset there are many botnets flows that resemble normal background traffic. Instead of labeling them as noise, DBSCAN allows them to blend in with the dense clusters.
Isolation Forest had high Accuracy (98.4%) but only detected 7.5% of botnet flows. It is more conservative than K-Means: rather than marking lots of points as anomalies, it chooses only a small subset which means fewer false positives but it also misses many attacks.
 
Figure 1. Elbow method and Silhouette score for k = 2..6.

 
Figure 2. Precision, Recall, and F1 across the three models
 
Figure 3. PCA projection with true labels (red = botnet)
5. Discussion
The chart above illustrates the classic Precision vs Recall trade-off. As far as security is concerned, failing to detect an actual botnet attack (False Negative) is worse than investigating a few extra flows (False Positive), so Recall/Precision trade-off is often weighted towards Recall. By this metric K-Means is the superior model out of the three we tested, despite its low Precision score.
Inspecting the K-Means clusters, we can see that the malicious cluster contains flows with small durations, small byte count, and repetitively contacted destination ports – behavior characteristic of Command and Control (C&C) communication. Some of the other clusters represent background TCP traffic, regular user behavior, and DNS-like requests.
Isolation Forest can therefore act as a second filter atop of the results from K-Means: if a particular flow is flagged by BOTH K-Means (as belonging to the suspicious cluster) and Isolation Forest (as an anomaly), it is likely to be a true positive attack. Using both would yield higher-confidence alerts.
The main limitation is that we used only basic features. A real system would also use temporal patterns (flow inter-arrival times), connection graphs, and reputation data for IP addresses and ports.
6. Conclusion
We ran K-Means, DBSCAN, and Isolation Forest on the CTU-13 dataset without utilizing labels during training. Recall (77.8%) was highest with K-Means using k=6 clusters, which was ideal since we want high Recall for an early warning system. DBSCAN did not perform well with this dataset and Isolation Forest was incredibly accurate but caught almost no attacks. The clustering was consistent with our expectations of botnet behavior. This indicates that unsupervised learning may have its place as the first layer of a SOC pipeline, even if it is unlabeled attack data.

CTU-13 dataset: https://www.stratosphereips.org/datasets-ctu13 
scikit-learn documentation: https://scikit-learn.org/stable/ 

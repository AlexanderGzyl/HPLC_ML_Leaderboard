# HPLC_ML_Leaderboard:
This repository serves as the central hub for the HPLC automation project's leaderboard. This project focuses on benchmarking machine learning models for predicting retention times in chromatographic enantiomer separation. The leaderboard provides a transparent way to monitor progress, compare different modeling approaches, and recognize top-performing contributions. Hosted on a Streamlit app, it will also offer researchers direct access to the models for making predictions, thereby accelerating the optimization of enantiomer separations. The top models will be capable of suggesting conditions most likely to yield successful results.

# Pitch: Unlock Faster, Smarter Chiral Separation Optimization with Machine Learning

Optimizing enantiomer separations using normal-phase chiral HPLC is notoriously complex and time-consuming. The traditional approach relies heavily on empirical trial-and-error, testing various combinations of stationary phases, mobile phase compositions, additives, and temperatures. This iterative process consumes valuable time, expensive chiral columns, and significant amounts of solvent, slowing down drug development and analysis.

**But what if you could move beyond intuition and guesswork?**

Machine learning offers a powerful, data-driven solution to this challenge. By training models on experimental separation data – including failed attempts and successful runs – we can capture the intricate, non-linear relationships between chromatographic conditions and separation outcomes (like resolution and retention).

**Imagine:**

Instead of randomly trying new conditions, you could predict the separation quality before running the experiment.
The model could suggest the most promising conditions to test, drastically reducing the number of experiments needed to find a baseline separation or even the optimal conditions.
You could explore the multi-dimensional experimental space much more efficiently, identifying optimal zones that might be missed by manual tuning.

# About The Dataset:
The leaderboard tracks the performance of different styles machine learning models on a HPLC retentention time dataset consisting of enantiomers. 
The dataset was created by Xu, H., Lin, J., Zhang, D. et al. (2023, [https://doi.org/10.1038/s41467-023-38853-3https://www.nature.com/articles/s41467-023-38853-3](https://doi.org/10.1038/s41467-023-38853-3https://www.nature.com/articles/s41467-023-38853-3)
The dataset constitutes the retention time of 25,847 molecules recorded in the form of SMILES, which contains 11,720 pairs of enantiomers,  polar modifier proportion, and chiral stationary phase. 



# Background:

According to the No Free Lunch Theorem (NFLT) in optimization and machine learning, when performance is averaged across the set of all possible problems, every algorithm performs equally well. More formally, for a uniform distribution over all possible problems, no algorithm is superior to any other on average. The practical consequence of this is profound: there is no single algorithm that is universally the best for every possible task. An algorithm that performs exceptionally well on one type of problem will, by necessity when averaged over all problems, perform poorly on others. This is precisely why benchmarking is vital when applying machine learning to a specific, real-world domain. Since real-world problems are not drawn uniformly from the set of all possible problems – they possess specific structures and characteristics – an algorithm's performance will vary depending on how well it is suited to that structure. The field of chemoinformatics serves as an excellent example of this. Researchers must make many choices regarding how to represent chemical information (descriptors) and which model architecture to use. This challenge is particularly evident in problems like enantiomer separation prediction, where diverse methods exist for describing the enantiomers and the stationary phase. To find the most effective approach for this specific task, empirical benchmarking on relevant data is indispensable, as the NFLT tells us we cannot assume a method that worked elsewhere will be optimal here.

Enantiomers, as defined by IUPAC, are a pair of molecular entities that stand as mirror images of each other and are non-superposable. These molecular twins share an identical connectivity of atoms within their structure but differ fundamentally in their three-dimensional arrangement. In typical laboratory settings, enantiomers exhibit nearly identical physical properties. Their melting points, boiling points, densities, and solubilities in achiral solvents are indistinguishable. However, a key difference arises in their interaction with plane-polarized light and in chiral environments.

The subtle difference in spatial arrangement between enantiomers has profound implications when they interact with chiral systems, including biological receptors and chiral stationary phases used in chromatography. This difference in interaction is the fundamental basis for enantioseparation techniques. The pharmaceutical industry, in particular, places immense importance on enantioseparation because the different enantiomers of a drug can exhibit drastically different pharmacological activities, toxicities, and metabolic pathways within the body. For instance, one enantiomer of a drug might possess the desired therapeutic effect, while the other could be inactive or even cause harmful side effects. Regulatory agencies often mandate the evaluation of each individual enantiomer of a chiral drug to ensure its safety and efficacy. The tragic case of thalidomide, where one enantiomer was a safe sedative while the other caused severe birth defects, serves as a critical example of the importance of enantiopurity in pharmaceuticals. The ability to obtain enantiomerically pure compounds is thus essential for advancing drug development and ensuring patient safety.

High-Performance Liquid Chromatography (HPLC) is a sophisticated analytical technique employed to separate the components of a sample that is dissolved in a liquid. The separation process relies on the principle of differential partitioning of the sample components between a mobile phase, which is a liquid solvent, and a stationary phase, which consists of a material packed into a column. The mobile phase acts as a carrier, transporting the sample through the column. As the sample components travel along the column, they interact with both the mobile and stationary phases. Components that exhibit a higher affinity for the stationary phase will be retained for a longer duration, while those with a greater affinity for the mobile phase will move through the column more rapidly. This difference in migration speed leads to the separation of the various components of the sample.

 An HPLC system typically comprises several key elements, including reservoirs containing the mobile phase, a high-pressure pump to ensure a constant flow of the mobile phase, an injector for introducing the sample into the system, the separation column packed with the stationary phase, and a detector to monitor the eluent as it exits the column. The detector provides a signal that is proportional to the amount of each separated component, allowing for qualitative and quantitative analysis of the sample.

 The effectiveness of an HPLC separation hinges on achieving a delicate balance of intermolecular forces among the three key players in the process: the solute (analyte), the mobile phase, and the stationary phase. The choice of the mobile and stationary phases must be carefully considered based on the properties of the analytes to be separated. The goal is to maximize the differences in the affinities of the various sample components for the stationary phase, thereby leading to their effective separation as they are carried by the mobile phase. The principle of "like attracts like" is often invoked to guide the selection of these phases. For instance, polar analytes will tend to interact more strongly with a polar stationary phase, while non-polar analytes will have a greater affinity for a non-polar stationary phase. Understanding the polarity and other chemical characteristics of the analytes is therefore crucial for developing an effective HPLC separation method.

Enantiomers, by virtue of their identical physical and chemical properties when in an achiral environment, cannot be distinguished or separated using conventional achiral HPLC columns. These standard columns typically separate molecules based on differences in properties like polarity or hydrophobicity, which are the same for enantiomers.

To achieve the separation of enantiomers, a chiral environment must be introduced into the chromatographic system. This is most commonly accomplished by employing a chiral stationary phase (CSP). In chiral HPLC, the CSP contains a chiral selector, which is an enantiomerically pure compound that can interact with the enantiomers of the analyte in a stereospecific manner. This interaction leads to the formation of transient diastereomeric complexes between the chiral selector and each of the enantiomers. Because the chiral selector is itself chiral, the interactions it forms with the two enantiomers of the analyte are not identical. These diastereomeric complexes have different energy levels and stabilities, resulting in different retention times for the two enantiomers as they pass through the column. This difference in retention allows for their separation.  Given the diversity of chiral compounds and the complexity of chiral recognition, it is often necessary to screen multiple CSPs with different chiral selectors to find the most suitable one for a particular enantioseparation. 

The composition of the mobile phase in normal phase HPLC has a significant impact on the chiral recognition process occurring on the CSP. In normal phase chiral HPLC, the mobile phase typically consists of a non-polar solvent, such as hexane or heptane, modified by the addition of a polar organic solvent, such as isopropanol, ethanol, or methanol. The type and the concentration of this polar modifier are critical parameters that can affect both the retention of the analytes and the degree of enantioselectivity achieved.

The polar modifier in the mobile phase can influence the interactions between the chiral analyte and the CSP in several ways. It can compete with the analyte for binding sites on the chiral selector, thereby altering the equilibrium of the formation of the diastereomeric complexes. The modifier can also interact with the chiral selector itself, potentially affecting its conformation or solvation and thus its ability to recognize the enantiomers of the analyte.



## **Environment setup**

```bash
conda env create -f environment.yml
conda activate hplc-ml
```





# 1: Introduction

- [1: Introduction](#1-introduction)
  - [What is Radiation Oncology?](#what-is-radiation-oncology)
    - [How is Treatment Planned?](#how-is-treatment-planned)
  - [A Brief History of AI](#a-brief-history-of-ai)
  - [AI In Radiation Oncology](#ai-in-radiation-oncology)
    - [Examples of AI in Action](#examples-of-ai-in-action)
  - [Types of Machine Learning](#types-of-machine-learning)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Semi-supervised and Self-supervised Learning](#semi-supervised-and-self-supervised-learning)
  - [Key Machine Learning Algorithms](#key-machine-learning-algorithms)
    - [Linear and Logistic Regression](#linear-and-logistic-regression)
    - [Decision Trees and Random Forests](#decision-trees-and-random-forests)
    - [Support Vector Machines](#support-vector-machines)
    - [K-means Clustering](#k-means-clustering)
  - [Challenges and Opportunities](#challenges-and-opportunities)
  - [Basics of Model Evaluation](#basics-of-model-evaluation)
    - [Train-Test Splits and Cross-Validation](#train-test-splits-and-cross-validation)
    - [Overfitting and Underfitting](#overfitting-and-underfitting)
    - [Performance Metrics](#performance-metrics)
  - [Limitations of Traditional Machine Learning](#limitations-of-traditional-machine-learning)
    - [Want to Learn More?](#want-to-learn-more)


## What is Radiation Oncology?

Radiation oncology is a key part of cancer treatment. In fact, more than half of all cancer patients receive radiotherapy at some point during their care [(Belanger et al., 2019)](https://iopscience.iop.org/article/10.1088/1361-6560/ab1817)[(Delaney et al., 2005)](https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.21324). This specialty stands out because it relies heavily on technology and computers, making it a natural fit for computational and data science methods [(Vogelius et al., 2020)](https://febs.onlinelibrary.wiley.com/doi/10.1002/1878-0261.12685).

Radiotherapy (RT) uses high-energy radiation like photons, electrons, or protons to damage the DNA of cancer cells. This stops them from growing and dividing [(Gao et al., 2024)](https://arxiv.org/abs/2406.01853). The main goal is simple: "treat the tumor while protecting healthy tissue." But achieving this is complex and requires advanced calculations and computer methods [(Barragan-Montero et al., 2022)](https://iopscience.iop.org/article/10.1088/1361-6560/ac678a/pdf).

There are different ways to deliver radiation therapy. The two main types are:
- **External Beam Radiation Therapy (EBRT):** Radiation comes from outside the body.
- **Internal Radiation Therapy (Brachytherapy):** Radiation is placed inside the body, near the tumor [(Belanger et al., 2019)](https://iopscience.iop.org/article/10.1088/1361-6560/ab1817).

Within EBRT, advanced techniques like Intensity-Modulated Radiation Therapy (IMRT) and Volumetric Modulated Arc Therapy (VMAT) allow for very precise targeting of tumors [(Gao et al., 2024)](https://arxiv.org/abs/2406.01853).

### How is Treatment Planned?

Planning radiation therapy involves defining exactly where to treat. Doctors use terms like **Planning Target Volume (PTV)**, which includes the tumor and a margin around it to account for movement or positioning errors [(Gao et al., 2024)](https://arxiv.org/abs/2406.01853). Planning is a mathematical optimization problem: deliver enough radiation to the tumor, but as little as possible to healthy organs [(Belanger et al., 2019)](https://iopscience.iop.org/article/10.1088/1361-6560/ab1817).

---

## A Brief History of AI

The early decades of AI research were characterized by periods of great optimism followed by "AI winters" when progress slowed and funding decreased. Early AI approaches focused on symbolic reasoning and rule-based systems, attempting to encode human knowledge explicitly. These systems showed promise in narrow domains but struggled with the complexity and ambiguity of real-world problems.

The evolution of AI has been marked by several paradigm shifts. From the rule-based expert systems of the 1970s and 1980s to the statistical approaches that gained prominence in the 1990s, each era brought new insights and techniques. The current deep learning revolution, which began in earnest around 2012 with breakthroughs in image recognition using convolutional neural networks, represents perhaps the most significant shift yet.

In radiation oncology, this evolution mirrors the broader field, with early applications focusing on rule-based planning systems, followed by statistical models for outcome prediction, and now deep learning approaches for tasks ranging from image segmentation to treatment planning optimization.

---

## AI In Radiation Oncology

Artificial Intelligence (AI) and Machine Learning (ML) are changing the way radiation oncology works. AI refers to computer systems that can perform tasks that usually require human intelligence, such as analyzing images or optimizing treatment plans [(Thomas et al., 2020)](https://pubmed.ncbi.nlm.nih.gov/32305726/). Because radiation oncology already relies on computers, it is well-positioned to benefit from AI [(Kiser et al., 2019)](https://jmai.amegroups.org/article/view/4996/html).

The radiotherapy process involves several steps: consultation, imaging, outlining the tumor (contouring), planning the treatment, quality checks, delivering the treatment, and follow-up [(Liu et al., 2023)](https://www.sciencedirect.com/science/article/pii/S2667005423000479). AI tools can help at almost every stage, especially in:
- **Image segmentation:** Automatically outlining tumors and organs
- **Treatment planning:** Creating and optimizing plans
- **Quality assurance:** Checking that plans are safe and effective
- **Outcome prediction:** Estimating how patients will respond [(Barragan-Montero et al., 2022)](https://iopscience.iop.org/article/10.1088/1361-6560/ac678a/pdf)

AI applications in radiation oncology can be broadly categorized into two key areas. First, in the practical aspects of radiation planning and delivery, where image analysis algorithms can automate tumor and normal tissue segmentation, and ML models can develop treatment plans based on prior cases. Second, these systems can provide quality assurance and remote review services for treatment plans [(Thomas et al., 2020)](https://pubmed.ncbi.nlm.nih.gov/32305726/). Deep learning approaches have demonstrated particular promise for contouring organs at risk (OARs) in head and neck cancer treatment planning, achieving high accuracy while reducing clinical workload [(Lastrucci et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11083654/) [(Liu et al., 2023)](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-023-01159-y).

### Examples of AI in Action

- **Contouring:** Deep learning can outline organs at risk (OARs) in head and neck cancer, saving time and improving accuracy [(Lastrucci et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11083654/) [(Liu et al., 2023)](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-023-01159-y).
- **Treatment Planning:** AI can automate the process, making it faster and more consistent [(Wang et al., 2019)](https://journals.sagepub.com/doi/pdf/10.1177/1533033819873922).
- **Adaptive Radiotherapy:** AI helps adjust plans in real time, for example, by quickly reconstructing images or generating CT-like images from MRI [(Fiorino et al., 2020)](https://febs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/1878-0261.12659).

Beyond these technical applications, AI has the potential to standardize and improve clinical practice by mitigating variability and suboptimality related to human factors [(Wang et al., 2020)](https://www.frontiersin.org/articles/10.3389/fonc.2020.580919/pdf). It can also facilitate knowledge transfer from more experienced to less experienced centers, helping disseminate expertise in planning new or emerging treatments and supporting radiation oncology practice in developing countries [(Barragan-Montero et al., 2022)](https://iopscience.iop.org/article/10.1088/1361-6560/ac678a/pdf).

---

## Types of Machine Learning

Machine learning, a subset of AI, focuses on developing algorithms that can learn patterns from data without being explicitly programmed. The field is typically divided into several learning paradigms:

### Supervised Learning

Supervised learning involves training models on labeled data, where each input is paired with the desired output. The algorithm learns to map inputs to outputs by minimizing the difference between its predictions and the ground truth labels. This approach is particularly relevant in radiation oncology for tasks like tumor classification, where historical images with confirmed diagnoses serve as training data.

In supervised learning, the quality and quantity of labeled data are crucial factors in model performance. For medical applications, obtaining high-quality labeled data often requires expert annotation, which can be time-consuming and expensive. This challenge is particularly acute in radiation oncology, where inter-observer variability in contouring can introduce inconsistencies in training data.

Common supervised learning tasks include classification (assigning inputs to discrete categories) and regression (predicting continuous values). In radiation oncology, classification might involve determining whether tissue is cancerous, while regression could predict radiation dose distribution.

### Unsupervised Learning

Unsupervised learning works with unlabeled data, seeking to discover inherent patterns or structures. Without explicit guidance on what constitutes a "correct" output, these algorithms identify natural groupings, reduce dimensionality, or detect anomalies in data.

Clustering algorithms, a major category of unsupervised learning, group similar data points together based on distance metrics. In radiation oncology, clustering might be used to identify patient subgroups with similar treatment responses or to detect patterns in treatment planning that correlate with outcomes.

Dimensionality reduction techniques like Principal Component Analysis (PCA) or t-SNE help visualize high-dimensional data and can reveal underlying structures not immediately apparent in the original feature space. These techniques can be valuable for analyzing the complex, multi-dimensional data generated in radiation treatment planning.

Anomaly detection algorithms identify outliers or unusual patterns in data, which could represent equipment malfunctions, unusual patient anatomy, or potential errors in treatment plans.

### Reinforcement Learning

Reinforcement learning involves training agents to make sequences of decisions by rewarding desired behaviors and penalizing undesired ones. Unlike supervised learning, there are no labeled examples; instead, the agent learns through trial and error, guided by a reward signal.

This paradigm has shown promise in treatment planning optimization, where the algorithm can learn to generate plans that maximize tumor coverage while minimizing dose to organs at risk. The reward function in such applications might incorporate clinical objectives like dose constraints and target coverage metrics.

Reinforcement learning faces challenges in medical applications due to the need for extensive exploration (trying different actions to learn their outcomes), which may not be feasible in clinical settings where patient safety is paramount. Simulation environments and digital twins offer potential solutions, allowing algorithms to learn in virtual environments before deployment in clinical practice.

### Semi-supervised and Self-supervised Learning

Semi-supervised learning combines elements of supervised and unsupervised learning, using a small amount of labeled data alongside a larger pool of unlabeled data. This approach is particularly relevant in medical imaging, where expert annotations may be limited but unannotated images are abundant.

Self-supervised learning, a growing area of research, involves creating supervised learning tasks from unlabeled data by generating labels automatically. For example, an algorithm might be trained to predict missing portions of an image, with the complete image serving as the ground truth. These approaches show promise for pre-training models when labeled data is scarce.

---

## Key Machine Learning Algorithms

Before diving into deep learning, it's essential to understand the traditional machine learning algorithms that form the foundation of the field. These algorithms continue to be valuable tools, especially when data is limited or interpretability is crucial.

### Linear and Logistic Regression

Linear regression, one of the simplest machine learning algorithms, models the relationship between input features and a continuous output variable as a linear function. Despite its simplicity, linear regression provides a foundation for understanding more complex models and can be surprisingly effective for certain problems.

Logistic regression extends this concept to classification problems by applying a sigmoid function to the linear output, transforming it into a probability between 0 and 1. This algorithm has been used in radiation oncology for predicting binary outcomes like tumor control or the development of specific toxicities.

Both linear and logistic regression offer high interpretability, as the contribution of each feature to the prediction is explicitly represented by its coefficient. This transparency is valuable in clinical settings where understanding the basis for predictions is essential.

### Decision Trees and Random Forests

Decision trees partition the feature space through a series of binary splits, creating a tree-like structure where each leaf node represents a prediction. These models are intuitive and can capture non-linear relationships, but individual trees are prone to overfitting.

Random forests address this limitation by combining many decision trees trained on different subsets of the data and features. The ensemble prediction is typically more robust and accurate than any individual tree. In radiation oncology, random forests have been used for tasks like predicting patient outcomes based on clinical factors, dosimetric parameters, and imaging features.

Tree-based methods offer several advantages for medical applications, including handling mixed data types, robustness to outliers, and the ability to capture complex interactions between features. They also provide measures of feature importance, helping identify the most relevant factors for a given prediction task.

### Support Vector Machines

Support Vector Machines (SVMs) find the optimal hyperplane that separates different classes in the feature space, maximizing the margin between the closest points (support vectors) from each class. Through the use of kernel functions, SVMs can efficiently handle non-linear decision boundaries.

SVMs have been applied in radiation oncology for tasks like classifying treatment outcomes based on dosimetric and clinical features. Their ability to work well with relatively small datasets makes them suitable for many medical applications where data may be limited.

### K-means Clustering

K-means clustering, an unsupervised learning algorithm, partitions data into K clusters by iteratively assigning points to the nearest cluster center and then updating those centers. This algorithm can identify natural groupings in patient data, potentially revealing subpopulations with distinct characteristics or treatment responses.

In radiation oncology, k-means clustering has been used to identify patient subgroups based on anatomical features, dose distributions, or treatment outcomes. These insights can inform personalized treatment approaches and help identify patients who might benefit from alternative strategies.

---

## Challenges and Opportunities

The integration of AI in radiation oncology extends beyond automating existing processes. It enables the analysis of multisource data that integrate variables from time-dependent sources, such as sequential quantitative imaging or genetic biomarkers. This capability could dramatically change radiotherapy approaches and play a central role in developing personalized, precision medicine [(Kazmierska et al., 2020)](https://www.thegreenjournal.com/article/S0167-8140(20)30829-X/pdf). 

The vision for AI in radiation oncology extends beyond targeted applications toward fully integrated data management systems with continuous feedback loops between patient outcomes and model inputs. This integration aims to improve clinical decision-making, enable accurate prediction of treatment outcomes and quality of life, and support efficient treatment planning and delivery [(Field et al., 2021)](https://pubmed.ncbi.nlm.nih.gov/34307915/). Recent innovations include foundation models like RO-LMM, which demonstrate proficiency across multiple tasks in the radiation oncology workflow, including clinical report summarization, treatment plan suggestion, and 3D target volume segmentation [(Kim et al., 2023)](https://arxiv.org/abs/2311.15876). The Radiation Oncology NLP Database (ROND) is the first dedicated NLP dataset for the specialty, which has historically received limited attention from the NLP community [(Liu et al., 2024)](https://arxiv.org/abs/2401.10995). As NLP technology matures, its integration into clinical settings can focus on high-priority tasks and contribute to assembling clinical corpora with appropriate guidelines and performance metrics [(Lin et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10874185/). 

Despite these advances, it's important to recognize that AI currently excels at replicating, automating, and standardizing human behavior on manual tasks, while conceptual clinical challenges related to definition, evaluation, and judgment remain in the realm of human intelligence [(Jarrett et al., 2019)](https://academic.oup.com/bjr/article/92/1100/20190001/7449195?login=true). Significant challenges therefore remain in implementing AI solutions in radiation oncology. These include the complexity of patient-specific disease characteristics, interplay with systemic therapies, inconsistent data recording methods, and limitations in treatment outcome reporting [(Field et al., 2021)](https://pubmed.ncbi.nlm.nih.gov/34307915/). Studies often lack information about the confidence levels associated with AI predictions and rarely assess how these technologies impact clinical outcomes [(Lastrucci et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11083654/)[(Franzese et al., 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10301548/). The full potential of AI in radiation oncology and medical physics will require addressing challenges related to data privacy, bias, and the continued need for human expertise [(Fionda et al., 2024)](https://pubmed.ncbi.nlm.nih.gov/39381628/)[(Alowais et al., 2023)](https://bmcmededuc.biomedcentral.com/articles/10.1186/s12909-023-04698-z).

However, AI is best at automating routine tasks. Complex clinical decisions still require human judgment [(Jarrett et al., 2019)](https://academic.oup.com/bjr/article/92/1100/20190001/7449195?login=true). There are also challenges:
- Patient differences and complex diseases
- Inconsistent data recording
- Data privacy and bias
- Limited information about how AI affects real patient outcomes [(Field et al., 2021)](https://pubmed.ncbi.nlm.nih.gov/34307915/)[(Lastrucci et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11083654/)[(Franzese et al., 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10301548/)

To succeed, AI in radiation oncology needs collaboration between doctors and computer scientists. This ensures that AI tools solve real clinical problems and meet high standards [(Lin et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10874185/)[(Lastrucci et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11083654/).

---

## Basics of Model Evaluation

It's important to know how well an AI model works before using it in practice. Here are some basics:

### Train-Test Splits and Cross-Validation

To test a model, data is split into training and testing sets. Cross-validation repeats this process several times for a more reliable result. In small medical datasets, special techniques make sure all important groups are included.

### Overfitting and Underfitting

- **Overfitting:** The model memorizes the training data but fails on new data.
- **Underfitting:** The model is too simple and misses important patterns.

Regularization and careful validation help avoid these problems.

### Performance Metrics

Different tasks need different ways to measure success:
- **Accuracy:** How often the model is right (good for balanced problems)
- **Precision & Recall:** Useful when one class is much rarer than another
- **F1 Score:** Balances precision and recall
- **AUC:** Measures how well the model separates classes
- **MSE/MAE:** For regression (predicting numbers)

In radiation oncology, special metrics are used, like Dice similarity for comparing shapes, or dose coverage for treatment plans.

---

## Limitations of Traditional Machine Learning

While traditional machine learning algorithms have proven valuable in many applications, they have several limitations that deep learning addresses:

1. **Feature engineering dependency**: Traditional algorithms rely heavily on manual feature engineering, which requires domain expertise and can miss complex patterns that aren't explicitly encoded.

2. **Difficulty with unstructured data**: Images, text, and other unstructured data types are challenging for traditional algorithms without extensive preprocessing.

3. **Limited representation capacity**: Many traditional algorithms struggle to capture complex, hierarchical patterns in data.

4. **Fixed model complexity**: The complexity of traditional models is often fixed or limited by design, constraining their ability to scale with data size.

5. **Separate learning stages**: Traditional pipelines often involve separate stages for feature extraction and model training, preventing end-to-end optimization.

Deep learning addresses these limitations through its ability to automatically learn hierarchical representations from raw data, scale with data and computational resources, and enable end-to-end training. However, traditional methods retain advantages in scenarios with limited data, when interpretability is crucial, or when computational resources are constrained.

Understanding these traditional approaches provides important context for appreciating the innovations and capabilities of deep learning, which we'll explore in subsequent modules. It also helps identify situations where simpler models might be more appropriate than complex deep learning architectures.

These limitations motivated the development of deep learning, which can automatically learn complex, hierarchical representations from raw data. In the next sections, we explore the building blocks and architectures that make deep learning so powerful for medical applications.

### Want to Learn More?

A [glossary](../_pages/glossary.md) of terms has been compiled to help demystify the acronyms and keywords in this field.

- [Artificial intelligence in radiation oncology](https://pubmed.ncbi.nlm.nih.gov/32843739/): A comprehensive overview of AI methods in radiation therapy.
- [Revolutionizing radiation therapy: the role of AI in clinical practice](https://academic.oup.com/jrr/article/65/1/1/7441099): How AI is optimizing tumor and organ segmentation.
- [Artificial intelligence in radiation oncology: A review of its current applications and future outlook](https://www.sciencedirect.com/science/article/pii/S1078817421000924): AI in toxicity prediction, automated planning, and more.
- [Artificial intelligence and machine learning in cancer imaging](https://www.nature.com/articles/s43856-022-00199-0): Challenges and opportunities for AI in cancer imaging.

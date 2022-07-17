
# Table of Contents



This repository contains code for the paper: ***LogGENE**: A smooth alternative to check loss for Deep Healthcare Inference Tasks* (pre-print can be found: [**HERE**](https://arxiv.org/abs/2206.09333))

The `/utils/` folder contains the code for the actual loss functions, with torch wrappers which allows for them to be invoked in your own work simply by doing: `cirterion=loss()` (where `loss` is `TiltedLossLC` if you want to use the tilted log cosh in a regression problem, `sBQRq` if you want to learn multiple quantiles in a binary classification problem and `sBQRl`, when you just want to learn a single quantile)

The Jupyter notebooks contain the code for the experiments (along with embedded results for the latest runs). The `GEOLoaderFinal.ipynb` notebook contains tests with the D-GEX dataset (which hasn&rsquo;t been provided with the repository because of it&rsquo;s size), `classification.ipynb` contains the code for testing the `sBQC` loss in datasets contained in `/Datasets/Classification/`, the `regression.ipynb` contains the code for testing the $\log\text{cosh}$ for the datasets contained in `/Datasets/Regression`.

Please feel free to open up an issue or make a pull request in case you find any inconsistencies or want to contribute!

BIBTeX citation:

> @misc{<https://doi.org/10.48550/arxiv.2206.09333>,
>   doi = {10.48550/ARXIV.2206.09333},
> 
> url = {<https://arxiv.org/abs/2206.09333>},
> 
> author = {Jeendgar, Aryaman and Pola, Aditya and Dhavala, Soma S and Saha, Snehanshu},
> 
> keywords = {Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
> 
> title = {LogGENE: A smooth alternative to check loss for Deep Healthcare Inference Tasks},
> 
> publisher = {arXiv},
> 
> year = {2022},
> 
>   copyright = {Creative Commons Attribution 4.0 International}
> }


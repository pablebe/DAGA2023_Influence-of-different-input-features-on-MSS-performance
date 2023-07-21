# Influence of different input features on musical source separation performance

This repository contains the code to train and evaluate different NN-based models for singing voice separation according to ([Bereuter, Sontacchi 2023](https://pub.dega-akustik.de/DAGA_2023/data/articles/000539.pdf)). The work was presented at the [DAGA2023](https://www.daga2023.de) in Hamburg.
All models are variants of the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) framework by ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)), with different modifications.
The carried out modifications comprise
- the data generation (model training with random mixing vs. without random mixing)
- the learning rate schedule (model training with LR schedule vs. without LR schedule)
- the spectral mask estimation (estimation of a magnitude spectral mask vs. complex valued mask)
- the loss function (training with magnitude MSE-loss vs. combined compressed spectral MSE-loss (CCMSE)) from ([Braun, Tashev, 2021](https://www.microsoft.com/en-us/research/uploads/prod/2021/08/23.pdf))
- the input features (magnitude spectral, complex valued spectral vs. Variable Q-Transform (VQT) features)

## MagSpec and MagSpecLog models

### Description

The folder `Training_MagSpec_based_open_unmix` contains python scripts for training and inference of Open-Unmix models operating on magnitude spectral and log magnitude spectral features. Variants of the Open-Unmix baseline model from ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)) with different training strategies e.g. with and without random mixing can be created.

For training of the MagSpec models the [MUSDB18-HQ](https://doi.org/10.5281/zenodo.3338373) dataset by ([Rafii, Liutkus, Stöter, Mimilakis, Bittner 2019](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav)) is used.

### Attributions and License

Within the folder `Training_MagSpec_based_open_unmix` large parts of the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) repository by ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)) were reused and modified, the corresponding software licencse of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) can be found within the subfolder `openunmix`.

## CompSpec models

### Description

The folder `Training_CompSpec_based_open_unmix` contains python scripts for training and inference of Open-Unmix models operating on complex-valued spectral features. The CompSpec models, their training scripts as well as their training strategy are strongly based on the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) framework.

For training of the CompSpec models the [MUSDB18-HQ](https://doi.org/10.5281/zenodo.3338373) dataset by ([Rafii, Liutkus, Stöter, Mimilakis, Bittner 2019](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav)) is used.

### Attributions and License

Within the folder `Training_CompSpec_based_open_unmix` large parts of the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) repository by ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)) were reused and modified, the corresponding software licencse of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) can be found within the subfolder `openunmix`.

## VQTSpec models

### Description

The folder `Training_VQTSpec_based_open_unmix` contains python scripts for training and inference of Open-Unmix models operating on Variable Q-Transform features. The VQTSpec models, their training scripts as well as their training strategy are strongly based on the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) framework. The Variable Q-Transform (VQT) is based on the Non-Stationary Gabor Transform (NSGT) by ([Holighaus, Dörfler, Velasco, Grill 2013](https://ieeexplore.ieee.org/document/6384709/)), where the python implementation by ([Grill 2011-2022](https://github.com/grrrr/nsgt)) is utilized. A similar model has also been proposed by ([Sevagh 2021](https://mdx-workshop.github.io/proceedings/hanssian.pdf)) using a slightly different filterbank for the calculation of the input features.

For training of the CompSpec models the [MUSDB18-HQ](https://doi.org/10.5281/zenodo.3338373) dataset by ([Rafii, Liutkus, Stöter, Mimilakis, Bittner 2019](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav)) is used and tuned to 440 Hz using the CQT-based Tuning algorithm proposed with ([Holzmüller, Bereuter, Merz, Rudrich, Sontacchi 2020](https://git.iem.at/audioplugins/cqt-analyzer)). The Matlab scripts for the dataset tuning can be found within the subfolder `Preprocessing_-studies_and_comparisons`.
### Attributions and License

Within the folder `Training_VQTSpec_based_open_unmix` large parts of the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) repository by ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)) as well as [XUMX-sliCQ](https://github.com/sevagh/xumx-sliCQ) were reused and modified, the corresponding software licencse of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) and [XUMX-sliCQ](https://github.com/sevagh/xumx-sliCQ) can be found within the subfolders `openunmix` and `openunmix_slicq`.

## Evaluation
The folder `Evaluation_open_unmix_models` contains all files to evaluate the trained models and plot the results as published in ([Bereuter & Sontacchi](https://pub.dega-akustik.de/DAGA_2023/data/articles/000539.pdf)) using the same metrics as for the [SISEC 2018](https://github.com/sigsep/sigsep-mus-2018), where the [BSS-Eval](https://sigsep.github.io/sigsep-mus-eval/) metrics by ([Vincent, Gribonval, Fevotte, 2006](https://ieeexplore.ieee.org/document/1643671/authors#authors)) are used to evaluate MSS performance.

Please note that for the evaluation of all models, the weights of the trained models are needed. The folder `Trained_models` contains all weights of the trained models, and it should be copied into the root folder of the repository (same level as `Evaluation_open_unmix_models`). The folder `Trained_models` can be downloaded in form of a *.zip*-file from the assets of the latest release.

## Audio Examples

Below you can listen to the results of 6 models trained with different input features and or loss functions, as presented in the results of ([Bereuter, Sontacchi 2023](https://pub.dega-akustik.de/DAGA_2023/data/articles/000539.pdf)).
The audio material is taken from the [MUSDB18-HQ](https://doi.org/10.5281/zenodo.3338373) test set (The Long Wait - Dark Horses).




https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/fe55e257-9969-46f3-a6aa-c4d4ff3e20d4

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/a9e5e8c9-808d-4055-a198-288e9439ae0a

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/89c1b31c-ca1b-448b-84ee-f5af8bc7c802

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/8c3a3906-a11e-4bda-9b52-eaf64bec1eb0

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/6ef363cd-27a2-4d37-ba84-8c3fb7ea1cf7

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/25e23315-b7a4-4e8d-8e3f-4df40b10f14e

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/82f3f781-4493-46fd-94ea-6ad6ce5bf9d2

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/9ec5ac66-6f90-433f-be85-eeaf02a66e62

https://github.com/pablebe/DAGA2023_Influence-of-different-input-features-on-MSS-performance/assets/58364449/62b10614-7cbc-471b-a8c9-07b2a9a22f56


## References

- F.-R. Stöter, S. Uhlich, A. Liutkus, and Y. Mitsufuji, "Open-unmix - a reference implementation for music source separation," Journal of Open Source Software, 2019, URL: https://joss.theoj.org/papers/10.21105/joss.01667
- S. Hanssian, "Music demixing with the slicq transform," MDX21 workshop at ISMIR 2021, [online] Available: https://mdx-workshop.github.io/proceedings/hanssian.pdf 
- N. Holighaus, M. Dörfler, G. A. Velasco, and T. Grill, "A framework for invertible, real-time constant-q transforms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 21, no. 4, pp. 775-785, 2013, [online] Available: https://ieeexplore.ieee.org/document/6384709/
- T. Grill, "Python implementation of non-stationary gabor transform (nsgt)," github repository, University of Vienna, Austria, 2011-2022. [accessed 23.06.23], [online] Available: https://github.com/grrrr/nsgt 
- F.-R. Stöter, A. Liutkus and N. Ito, "The 2018 signal separation evaluation campaign," in Latent Variable Analysis and Signal Separation, pp. 293-30, 2018, [online] Available: https://arxiv.org/pdf/1804.06267.pdf 
- S. Braun and I. Tashev, "A consolidated view of loss functions for supervised deep learning-based speech enhancement," 44th International Conference on Telecommunications and Signal Processing (TSP), pp. 72-76, 2021, [online] Available: https://arxiv.org/pdf/2009.12286.pdf 
- Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis and R. Bittner, "The MUSDB18 corpus for music separation," Dec. 2017, [online] Available: https://zenodo.org/record/3338373 
- E. Vincent, R. Gribonval and C. Fevotte, "Performance measurement in blind audio source separation," IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 4, pp. 1462-1469, 2006, [online] Available: https://ieeexplore.ieee.org/document/1643671/authors#authors 

## Citation

If you want to reuse any part of this code, which has not been published with any of the aforementioned refrences, we would appreciate if you'd cite our work with:

- P. A. Bereuter, A. Sontacchi, "Influence of different input features on musical source separation performance, " Proceedings 49th Annual German Conference on Acoustics (DAGA 2023), Fortschritte der Akustik - DAGA 2023, Deutsche Gesellschaft für Akustik e.V. (DEGA), vol. 49, pp 430-433, March 2023, [online] Available: https://pub.dega-akustik.de/DAGA_2023/data/daga23_proceedings.pdf 

## License
The licenses of all the reused third party code are enclosed in the corresponding subfolders.
The newly created content is distributed under the MIT License. See `LICENSE.txt` for more information.

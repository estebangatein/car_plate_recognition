# Car Plate Recognition and Reconstruction with Deep Learning

This project, realized in the context of the Computer Vision subject from the Artificial Intelligence and Robotics master at Sapienza, has the goal of detecting a car plate and extracting its plate number. The activities of the project consist of creating a baseline to solve the task and comparing the results to a replica of the proposed model from the next publication (referred to as _paper_ in the code and file names):

[1] Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791.

The dataset used is the [Chinese City Parking Dataset (CCPD)](https://github.com/detectRecog/CCPD), presented with more detail in the following publication:

[2] Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L. Towards end-to-end license plate
detection and recognition: A large dataset and baseline. In Proceedings of the European Conference on
Computer Vision (ECCV), Munich, Germany, 8–14 September 2018.

This repository includes the following files:
- `evaluation.ipynb`: notebook including the evaluation code of the models
- `paper_yolo_trainer.ipynb` & `paper_pdlpr_trainer.ipynb`: implementation of the models proposed in [1]
- `baseline_trainer.ipynb`: implementation of the baseline model solving the task
- `utils_esteban.py`: contains the models and data loaders for the evaluation notebook
- `model_weights`: directory including the weights of the trained models

Note that the notebooks don't include the outputs of the trainings, as they were trained on Kaggle using its available GPU runtime, and cell outputs don't persist after notebook execution ends. Also, the models were trained with the _weather_ partition of the dataset, as the _base_ one was too heavy, and evaluated on an OOD subset from the _base_ partition. Finally, the evaluation notebook can't be fully executed, as the weights of the PDLPR model aren't provided, the weights file is too large for the repository, and results are poor. If the file is really needed, send a mail to [esteban.gatein@gmail.com](mailto:esteban.gatein@gmail.com).
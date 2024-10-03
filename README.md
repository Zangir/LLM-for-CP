Official Implementation for the "Self-Guiding Exploration for Combinatorial Problems" paper. Accepted at NeurIPS 2024.

<p align="center">
    🌐 📃 <a href="https://arxiv.org/abs/2405.17950" target="_blank">Paper</a> <br>
</p>

**Authors**: [Zangir Iklassov](https://scholar.google.com/citations?user=SuLVY5oAAAAJ)<sup>[:email:️](mailto:zangir.iklassov@mbzuai.ac.ae)</sup>, Yali Du, Farkhad Akimov, [Martin Takac](https://mtakac.com/), *MBZUAI, United Arab Emirates*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**

## Code
### Required:
    Python==3.9+
    numpy==1.26.4
    scipy==1.9.1
    google-generativeai==0.4.0
    openai==1.13.3
    transformers==4.38.1
    datasets==2.17.1
    ortools==9.9.3963
    collections
    re
    os
    time

### TODO
A new version of the algorithm has been added to main.py. 
This version is more stable and robust, particularly for large instances of the Traveling Salesman Problem (TSP). 
It now generates Python code for each solution trajectory, rather than relying solely on the LLM. 
In future updates, other Combinatorial Problem (CP) tasks and Reasoning tasks will also be integrated into this version.

## Citation
If you find the code useful for your research or any other project, please consider citing our paper. :smiley:
```
@misc{iklassov2024selfguidingexplorationcombinatorialproblems,
      title={Self-Guiding Exploration for Combinatorial Problems}, 
      author={Zangir Iklassov and Yali Du and Farkhad Akimov and Martin Takac},
      year={2024},
      eprint={2405.17950},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.17950}, 
}

```

## Contact
If you meet any problems, please describe them in issues or contact:
* Zangir Iklassov: <zangir.iklassov@mbzuai.ac.ae>
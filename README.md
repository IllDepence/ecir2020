# Semantic Modelling of Citation Contexts for Context-aware Citation Recommendation
Source code and evaluation details for the ECIR 2020 paper [*Semantic Modelling of Citation Contexts for Context-aware Citation Recommendation*](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_15).

### Overview
* Semantic models based on claims and entities
    * Claim structures based on Universal Dependencies
    * Entities based on Noun Phrases
* Evaluated on four data sets
    * [unarXive](https://github.com/IllDepence/unarXive/)
    * [MAG](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/)
    * [RefSeer](https://psu.app.box.com/v/refseer)
    * [ACL-ARC](https://acl-arc.comp.nus.edu.sg/)
* Evaluation settings
    * Offline (citation re-prediction)
        * code: `code_offline_eval/`
        * data: [10.5281/zenodo.6243291](https://doi.org/10.5281/zenodo.6243291)
        * details: `eval/offline_eval*`
    * User study
        * code: `code_user_study/`
        * details: `eval/user_study*`

### Cite as

```
@inproceedings{Saier2020ECIR,
  author        = {Tarek Saier and
                   Michael F{\"{a}}rber},
  title         = {{Semantic Modelling of Citation Contexts for Context-aware Citation Recommendation}},
  booktitle     = {Proceedings of the 42nd European Conference on Information Retrieval},
  pages         = {220--233},
  year          = {2020},
  month         = apr,
  doi           = {10.1007/978-3-030-45439-5_15},
}
```


# The Abstraction and Reasoning Corpus (ARC)

### Tackling the ARC problem

General pipeline and future ideas:
1) Load data
2) Categorize by dimension:
    * 1:1 in:out
    * Linear dimension predictor
    * Custom dimension out based on image
3) Split up problems into buckets
4) Find most correlated pixels
5) Create best features for each bucket
6) Make models for each bucket

Completed prediction attempt in `pred.py`:
* A simple input - output model using Trees. Bucket: dimension 1:1, 10% Accuracy.